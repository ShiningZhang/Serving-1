// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/general-server/op/general_preprocess_op.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include "core/predictor/framework/infer.h"
#include "core/predictor/framework/memory.h"
#include "core/predictor/framework/resource.h"
#include "core/util/include/timer.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::Timer;
using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::predictor::general_model::Tensor;
using baidu::paddle_serving::predictor::general_model::Response;
using baidu::paddle_serving::predictor::general_model::Request;
using baidu::paddle_serving::predictor::InferManager;
using baidu::paddle_serving::predictor::PaddleGeneralModelConfig;


static int python_process(const paddle::PaddleTensor& input, paddle::PaddleTensor& output){
  std::string name = input.name;
  std::vector<std::vector<size_t>> lod = input.lod;
  std::vector<int> shape = input.shape;
  void* origin_data = input.data.data();
  paddle::PaddleDType type = input.dtype;

  py::array_t<uint8_t> npInputArray(shape, (const unsigned char*)origin_data);

  static py::module calc = py::module::import("preprocess");
  static auto func = calc.attr("preprocess");

  py::object result;
  try {
    result = func(npInputArray);
  } catch (std::exception& e) {
    std::cout << "call python func failed:" << e.what() << std::endl;;
    return false;
  }

  py::array_t<float> outArray = result.cast<py::array_t<float>>();
 
  // copy output data
  py::buffer_info outBuf = outArray.request();
  float* optr = (float*)outBuf.ptr;
  py::buffer_info buf1 = outArray.request();
  int dim_size = buf1.ndim;
  std::vector<int> out_shape(1,1);
  for (int i = 0; i < dim_size; ++i){
    out_shape.push_back(buf1.shape[i]);
  }
  // std::cout << "outArray.size()="<<outArray.size();

  output.dtype = paddle::PaddleDType::FLOAT32;
  output.lod = lod;
  output.shape = out_shape;
  output.name = name;
  paddle::PaddleBuf paddleBuf(outArray.size() * sizeof(float));

  memcpy(paddleBuf.data(), optr, outArray.size() * sizeof(float));
  output.data = paddleBuf;

  return 0;
}

int GeneralPreProcessOp::inference() {
  VLOG(2) << "Going to run inference";
  const std::vector<std::string> pre_node_names = pre_names();
  if (pre_node_names.size() != 1) {
    LOG(ERROR) << "This op(" << op_name()
               << ") can only have one predecessor op, but received "
               << pre_node_names.size();
    return -1;
  }
  const std::string pre_name = pre_node_names[0];

  const GeneralBlob *input_blob = get_depend_argument<GeneralBlob>(pre_name);
  if (!input_blob) {
    LOG(ERROR) << "input_blob is nullptr,error";
    return -1;
  }
  uint64_t log_id = input_blob->GetLogId();
  VLOG(2) << "(logid=" << log_id << ") Get precedent op name: " << pre_name;

  GeneralBlob *output_blob = mutable_data<GeneralBlob>();
  if (!output_blob) {
    LOG(ERROR) << "output_blob is nullptr,error";
    return -1;
  }
  output_blob->SetLogId(log_id);

  if (!input_blob) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed mutable depended argument, op:" << pre_name;
    return -1;
  }

  const TensorVector *in = &input_blob->tensor_vector;
  TensorVector *out = &output_blob->tensor_vector;

  int batch_size = input_blob->_batch_size;
  output_blob->_batch_size = batch_size;
  VLOG(2) << "(logid=" << log_id << ") infer batch size: " << batch_size;

  Timer timeline;
  int64_t start = timeline.TimeStampUS();
  timeline.Start();

  for (auto& val : *in){
    paddle::PaddleTensor output;
    static py::scoped_interpreter guard{};
    {
      python_process(val, output);
    }
    out->push_back(output);
  }
  
  int64_t end = timeline.TimeStampUS();
  CopyBlobInfo(input_blob, output_blob);
  AddBlobInfo(output_blob, start);
  AddBlobInfo(output_blob, end);
  return 0;
}
DEFINE_OP(GeneralPreProcessOp);

}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
