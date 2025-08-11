use crate::CpuStorageSlice;
use crate::Device;
use crate::Dim;
use crate::GGmlType;
use crate::RawSlice;
use crate::Storage;
use crate::StorageProto;
use crate::Tensor;
use crate::TensorProto;
use serde::de::{self, Deserializer};
use serde::ser::{SerializeSeq, Serializer};
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::ptr::NonNull;

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // 1. 先拿到一个 serde_json::Value
        let v = Value::deserialize(deserializer)
            .map_err(|e| de::Error::custom(format!("Failed to parse JSON Value: {}", e)))?;

        // 2. 递归遍历，收集 shapes 和 flat data
        let mut shapes: Vec<usize> = Vec::new();
        let mut flat: Vec<TensorElement> = Vec::new();

        fn traverse(
            val: &Value,
            depth: usize,
            shapes: &mut Vec<usize>,
            flat: &mut Vec<TensorElement>,
        ) -> Result<(), String> {
            match val {
                Value::Array(arr) => {
                    // 记录或校验这一层的长度
                    if shapes.len() <= depth {
                        shapes.push(arr.len());
                    } else if shapes[depth] != arr.len() {
                        return Err(format!(
                            "Inconsistent length at dim {}: expected {}, got {}",
                            depth,
                            shapes[depth],
                            arr.len()
                        ));
                    }
                    // 递归
                    for elem in arr {
                        traverse(elem, depth + 1, shapes, flat)?;
                    }
                }
                Value::Number(num) => {
                    if let Some(f) = num.as_f64() {
                        flat.push(TensorElement::F32(f as f32));
                    } else if let Some(i) = num.as_i64() {
                        flat.push(TensorElement::I32(i as i32));
                    } else {
                        return Err("Unsupported number type".to_string());
                    }
                }
                other => {
                    return Err(format!("Unexpected JSON at depth {}: {:?}", depth, other));
                }
            }
            Ok(())
        }

        traverse(&v, 0, &mut shapes, &mut flat)
            .map_err(|e| de::Error::custom(format!("Tensor traverse error: {}", e)))?;

        // 3. Shape must be non-empty
        if shapes.is_empty() {
            return Err(de::Error::custom(
                "Cannot deserialize a zero-dimensional (empty) tensor",
            ));
        }

        // 4. 推断 dtype
        let dtype = match flat.first() {
            Some(TensorElement::F32(_)) => GGmlType::F32,
            Some(TensorElement::I32(_)) => GGmlType::I32,
            None => return Err(de::Error::custom("Empty tensor is not supported")),
        };

        // 5. 校验总长度
        let total: usize = shapes.iter().product();
        if total != flat.len() {
            return Err(de::Error::custom(format!(
                "Element count {} but shape {:?} expects {}",
                flat.len(),
                shapes,
                total
            )));
        }
        let dim = Dim::from_vec(shapes, dtype);
        let dev = Device::Cpu;
        let data = match dtype {
            GGmlType::F32 => {
                let mut backing: Vec<f32> = flat.into_iter().map(|e| e.as_f32()).collect();
                let cap = backing.capacity();
                let ptr = backing.as_mut_ptr();
                std::mem::forget(backing);
                let raw = RawSlice {
                    ptr: unsafe { NonNull::new_unchecked(ptr) },
                    len: total,
                    cap,
                };
                Storage::Cpu(CpuStorageSlice::F32(raw))
            }
            GGmlType::I32 => {
                let mut backing: Vec<i32> = flat
                    .into_iter()
                    .map(|e| match e {
                        TensorElement::I32(i) => i,
                        TensorElement::F32(f) => f as i32,
                    })
                    .collect();
                let cap = backing.capacity();
                let ptr = backing.as_mut_ptr();
                std::mem::forget(backing);
                let raw = RawSlice {
                    ptr: unsafe { NonNull::new_unchecked(ptr) },
                    len: total,
                    cap,
                };
                Storage::Cpu(CpuStorageSlice::I32(raw))
            }
            _ => {
                return Err(de::Error::custom(
                    "Only f32 and i32 CPU tensors are supported",
                ))
            }
        };

        Ok(Tensor {
            dtype,
            data,
            dim,
            dev,
        })
    }
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        //人类可读的
        // if serializer.is_human_readable() {
        // 1. 提取原始数据为统一格式
        let (flat_data, elem_type) = match &self.data {
            Storage::Cpu(cpu_storage) => match cpu_storage {
                CpuStorageSlice::F32(raw_slice) => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(raw_slice.ptr.as_ptr(), raw_slice.len)
                    };
                    (
                        slice
                            .iter()
                            .map(|&x| TensorElement::F32(x))
                            .collect::<Vec<_>>(),
                        "f32",
                    )
                }
                CpuStorageSlice::I32(raw_slice) => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(raw_slice.ptr.as_ptr(), raw_slice.len)
                    };
                    (
                        slice.iter().map(|&x| TensorElement::I32(x)).collect(),
                        "i32",
                    )
                }
                _ => return Err(serde::ser::Error::custom("Unsupported storage type")),
            },
            #[cfg(feature = "cuda")]
            Storage::Gpu(_) => return Err(serde::ser::Error::custom("GPU storage not supported")),
        };

        // 2. 验证数据与形状匹配
        let total_elements: usize = self.dim.shape.0[..self.dim.n_dims].iter().product();
        if flat_data.len() != total_elements {
            return Err(serde::ser::Error::custom(format!(
                "Data length {} doesn't match shape {:?}",
                flat_data.len(),
                &self.dim.shape.0[..self.dim.n_dims]
            )));
        }

        // 3. 递归序列化为嵌套结构
        serialize_nested(&flat_data, &self.dim.shape.0[..self.dim.n_dims], serializer)
        // } else {
        //     let state = TensorStorageFormat {
        //         dtype: self.dtype.as_u8(),
        //         shape: self.shape(),
        //         data: self.data.as_bytes(),
        //     };
        //     state.serialize(serializer)
        // }
    }
}

// struct TensorStorageFormat<'a> {
//     dtype: u8,
//     shape: &'a [usize],
//     data: &'a [u8],
// }

// impl<'a> Serialize for TensorStorageFormat<'a> {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//     }
// }

#[derive(Debug, Deserialize)]
enum TensorElement {
    F32(f32),
    I32(i32),
}

impl TensorElement {
    fn as_f32(self) -> f32 {
        match self {
            TensorElement::I32(v) => v as f32,
            TensorElement::F32(v) => v,
        }
    }
}

impl Serialize for TensorElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TensorElement::F32(f) => serializer.serialize_f32(*f),
            TensorElement::I32(i) => serializer.serialize_i32(*i),
        }
    }
}

fn serialize_nested<S>(
    data: &[TensorElement],
    shape: &[usize],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match shape {
        // 标量（0维张量）
        [] => data[0].serialize(serializer),
        // 最内层序列
        [dim] => {
            let mut seq = serializer.serialize_seq(Some(*dim))?;
            for item in &data[..*dim] {
                seq.serialize_element(item)?;
            }
            seq.end()
        }
        // 递归处理高维
        [first, rest @ ..] => {
            let sub_len: usize = rest.iter().product();
            let mut seq = serializer.serialize_seq(Some(*first))?;
            for i in 0..*first {
                let start = i * sub_len;
                let end = start + sub_len;
                seq.serialize_element(&NestedArray {
                    data: &data[start..end],
                    shape: rest,
                })?;
            }
            seq.end()
        }
    }
}
struct NestedArray<'a> {
    data: &'a [TensorElement],
    shape: &'a [usize],
}

impl<'a> Serialize for NestedArray<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_nested(self.data, self.shape, serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助函数：提取 CPU-F32 存储里的数据为 Vec<f32>
    fn extract_f32(t: &Tensor) -> Vec<f32> {
        if let Storage::Cpu(CpuStorageSlice::F32(raw)) = &t.data {
            unsafe { std::slice::from_raw_parts(raw.ptr.as_ptr(), raw.len).to_vec() }
        } else {
            panic!("Expected F32 CPU storage");
        }
    }

    #[test]
    fn test_deserialize_2d_f32() {
        let json = "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]";
        let t: Tensor = serde_json::from_str(json).unwrap();
        println!("{:?}", t.dim);
        let data = extract_f32(&t);
        println!("data:{:?}", data);
    }

    #[test]
    fn test_tensor_serialize() {
        let m = Tensor::mat_slice(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu).unwrap();
        println!("{:?}", m.dim);
        let serialized = serde_json::to_value(&m).unwrap();
        println!("{}", serialized);
    }
}
