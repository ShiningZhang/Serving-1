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

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "core/general-client/include/grpc_client.h"
#include "core/sdk-cpp/include/common.h"
#include "core/util/include/timer.h"
#include "core/sdk-cpp/builtin_format.pb.h"
#include "core/sdk-cpp/general_model_service.pb.h"
#include "core/sdk-cpp/general_model_service.grpc.pb.h"
DECLARE_bool(profile_client);
DECLARE_bool(profile_server);
#define GRPC_MAX_BODY_SIZE 512 * 1024 * 1024

namespace baidu {
namespace paddle_serving {
namespace client {

using baidu::paddle_serving::Timer;
using baidu::paddle_serving::predictor::general_model::Request;
using baidu::paddle_serving::predictor::general_model::Response;
using baidu::paddle_serving::predictor::general_model::Tensor;
using baidu::paddle_serving::predictor::general_model::Tensor;
using baidu::paddle_serving::predictor::general_model::GeneralModelService;

using configure::SDKConf;
using configure::VariantConf;
using configure::Predictor;
using configure::VariantConf;

std::shared_ptr<grpc::Channel>
GetChannelStub(
    const std::string& url, const GrpcOptions& grpc_options) {
  // map<url, Channel*> used to keep track of GRPC channels.
  // reuse the established Channel of same url
  static std::map<std::string, std::shared_ptr<grpc::Channel>> stub_map;
  static std::mutex stub_map_mtx;

  std::lock_guard<std::mutex> lock(stub_map_mtx);
  const auto& channel_itr = stub_map.find(url);
  if (channel_itr != stub_map.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(GRPC_MAX_BODY_SIZE);
    arguments.SetMaxReceiveMessageSize(GRPC_MAX_BODY_SIZE);
    // GRPC KeepAlive: https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    arguments.SetInt(
        GRPC_ARG_KEEPALIVE_TIME_MS, grpc_options.keepalive_time_ms);
    arguments.SetInt(
        GRPC_ARG_KEEPALIVE_TIMEOUT_MS, grpc_options.keepalive_timeout_ms);
    arguments.SetInt(
        GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
        grpc_options.keepalive_permit_without_calls);
    arguments.SetInt(
        GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA,
        grpc_options.http2_max_pings_without_data);
    std::shared_ptr<grpc::ChannelCredentials> credentials =
        grpc::InsecureChannelCredentials();
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateCustomChannel(url, credentials, arguments);
    stub_map.insert(
        std::make_pair(url, channel));
    return channel;
  }
}

int ServingGrpcClient::connect(const std::string& server_port) {
  std::shared_ptr<grpc::Channel> channel = 
      GetChannelStub(server_port, GrpcOptions());
  stub_ = GeneralModelService::NewStub(channel);
  return 0;
}

int ServingGrpcClient::predict(const PredictorInputs& inputs,
                               PredictorOutputs& outputs,
                               const std::vector<std::string>& fetch_name,
                               const uint64_t log_id) {
  Timer timeline;
  int64_t preprocess_start = timeline.TimeStampUS();
  VLOG(2) << "max body size : " << GRPC_MAX_BODY_SIZE;
  Request req;
  req.set_log_id(log_id);
  for (auto &name : fetch_name) {
    req.add_fetch_var_names(name);
  }

  if (PredictorInputs::GenProto(inputs, _feed_name_to_idx, _feed_name, req) != 0) {
    LOG(ERROR) << "Failed to preprocess req!";
    return -1;
  }

  int64_t preprocess_end = timeline.TimeStampUS();
  int64_t client_infer_start = timeline.TimeStampUS();
  Response res;

  int64_t client_infer_end = 0;
  int64_t postprocess_start = 0;
  int64_t postprocess_end = 0;

  if (FLAGS_profile_client) {
    if (FLAGS_profile_server) {
      req.set_profile_server(true);
    }
  }

  res.Clear();
  grpc::ClientContext context;
  grpc::Status status = stub_->inference(&context, req, &res);
  if (!status.ok()) {
    LOG(ERROR) << "failed call predictor with req: " << req.ShortDebugString();
    return -1;
  }

  client_infer_end = timeline.TimeStampUS();
  postprocess_start = client_infer_end;
  if (PredictorOutputs::ParseProto(res, fetch_name, _fetch_name_to_type, outputs) != 0) {
    LOG(ERROR) << "Failed to post_process res!";
    return -1;
  }
  postprocess_end = timeline.TimeStampUS();

  if (FLAGS_profile_client) {
    std::ostringstream oss;
    oss << "PROFILE\t"
        << "pid:" << getpid() << "\t"
        << "prepro_0:" << preprocess_start << " "
        << "prepro_1:" << preprocess_end << " "
        << "client_infer_0:" << client_infer_start << " "
        << "client_infer_1:" << client_infer_end << " ";
    if (FLAGS_profile_server) {
      int op_num = res.profile_time_size() / 2;
      for (int i = 0; i < op_num; ++i) {
        oss << "op" << i << "_0:" << res.profile_time(i * 2) << " ";
        oss << "op" << i << "_1:" << res.profile_time(i * 2 + 1) << " ";
      }
    }

    oss << "postpro_0:" << postprocess_start << " ";
    oss << "postpro_1:" << postprocess_end;

    fprintf(stderr, "%s\n", oss.str().c_str());
  }

  return 0;
}

}  // namespace general_model
}  // namespace paddle_serving
}  // namespace baidu
