// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once
#include <memory>
#include "google/protobuf/service.h"
#include "core/general-client/include/client.h"
#include "core/sdk-cpp/general_model_service.grpc.pb.h"

namespace baidu {
namespace paddle_serving {
namespace client {

struct GrpcOptions {
  explicit GrpcOptions()
      : keepalive_time_ms(INT_MAX), keepalive_timeout_ms(20000),
        keepalive_permit_without_calls(false), http2_max_pings_without_data(2)
  {
  }
  // GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
  // The period (in milliseconds) after which a keepalive ping is sent on the
  // transport
  int keepalive_time_ms;
  // The amount of time (in milliseconds) the sender of the keepalive ping waits
  // for an acknowledgement. If it does not receive an acknowledgment within
  // this time, it will close the connection.
  int keepalive_timeout_ms;
  // If true, allow keepalive pings to be sent even if there are no calls in
  // flight.
  bool keepalive_permit_without_calls;
  // The maximum number of pings that can be sent when there is no data/header
  // frame to be sent. gRPC Core will not continue sending pings if we run over
  // the limit. Setting it to 0 allows sending pings without such a restriction.
  int http2_max_pings_without_data;
};

class ServingGrpcClient : public ServingClient {
 public:

  ServingGrpcClient() {};

  ~ServingGrpcClient() {};

  virtual int connect(const std::string& server_port);

  int predict(const PredictorInputs& inputs,
              PredictorOutputs& outputs,
              const std::vector<std::string>& fetch_name,
              const uint64_t log_id);

 private:
  std::shared_ptr<baidu::paddle_serving::predictor::general_model::
                      GeneralModelService::Stub>
      stub_;
};

}  // namespace client
}  // namespace paddle_serving
}  // namespace baidu