// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <vector>
#include <thread>
#include <mutex>

#include "core/general-client/include/brpc_client.h"
#include "core/util/include/timer.h"

using namespace std;  // NOLINT

using baidu::paddle_serving::client::ServingClient;
using baidu::paddle_serving::client::ServingBrpcClient;
using baidu::paddle_serving::client::PredictorInputs;
using baidu::paddle_serving::client::PredictorOutputs;
using baidu::paddle_serving::Timer;

DEFINE_string(server_port, "127.0.0.1:9292", "");
DEFINE_string(client_conf, "serving_client_conf.prototxt", "");
DEFINE_string(test_type, "brpc", "");
DEFINE_string(sample_type, "fit_a_line", "");
DEFINE_int32(test_thread_num, 1, "");
DEFINE_int32(test_count, 1, "");

namespace {
int prepare_fit_a_line(PredictorInputs& input, std::vector<std::string>& fetch_name, int batch_size = 1) {
  std::vector<float> float_feed = {0.0137f, -0.1136f, 0.2553f, -0.0692f,
            0.0582f, -0.0727f, -0.1583f, -0.0584f,
            0.6283f, 0.4919f, 0.1856f, 0.0795f, -0.0332f};
  std::vector<float> float_feed_batch;
  for (int i = 0; i < batch_size; ++i) {
    float_feed_batch.insert(float_feed_batch.end(), float_feed.begin(), float_feed.end());
  }
  std::vector<int> float_shape = {batch_size, 13};
  std::string feed_name = "x";
  fetch_name = {"price"};
  std::vector<int> lod;
  input.add_float_data(float_feed_batch, feed_name, float_shape, lod);
  return 0;
}

int prepare_bert(PredictorInputs& input, std::vector<std::string>& fetch_name) {
  float input_mask[] = {
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  long position_ids[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0};
  long input_ids[] = {
      101, 0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0,    0, 0,
      0,   0,    0,    0,   0,    0,    0,    0,    0,    0,    0};
  long segment_ids[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  {
    std::vector<float> float_feed(std::begin(input_mask), std::end(input_mask));
    std::vector<int> float_shape = {1, 128, 1};
    std::string feed_name = "input_mask";
    std::vector<int> lod;
    input.add_float_data(float_feed, feed_name, float_shape, lod);
  }
  {
    std::vector<int64_t> feed(std::begin(position_ids), std::end(position_ids));
    std::vector<int> shape = {1, 128, 1};
    std::string feed_name = "position_ids";
    std::vector<int> lod;
    input.add_int64_data(feed, feed_name, shape, lod);
  }
  {
    std::vector<int64_t> feed(std::begin(input_ids), std::end(input_ids));
    std::vector<int> shape = {1, 128, 1};
    std::string feed_name = "input_ids";
    std::vector<int> lod;
    input.add_int64_data(feed, feed_name, shape, lod);
  }
  {
    std::vector<int64_t> feed(std::begin(segment_ids), std::end(segment_ids));
    std::vector<int> shape = {1, 128, 1};
    std::string feed_name = "segment_ids";
    std::vector<int> lod;
    input.add_int64_data(feed, feed_name, shape, lod);
  }
  
  fetch_name = {"pooled_output"};
  return 0;
}

double total_thread_cost = 0;
int total_thread_count = 0;
int batch_size = 1;
int total_count = 0;
std::mutex g_mutex;

void thread_func(ServingClient* client, PredictorInputs input, std::vector<std::string> fetch_name, uint64_t log_id) {
  Timer timeline;
  timeline.Start();
  int count = 0;
  PredictorOutputs output;
  for (int i = 0; i < FLAGS_test_count; ++i) {
    client->predict(input, output, fetch_name, log_id);
    // LOG(INFO) << output.print();
    output.clear();
  }
  timeline.Pause();
  double cost = timeline.ElapsedMS();
  LOG(INFO) << "thread[" << log_id << "]:"
            << "total_cost=" << cost << "ms";
  std::lock_guard<std::mutex> lck(g_mutex);
  total_thread_cost += cost;
  total_thread_count++;
  total_count += FLAGS_test_count;
}
} // namespace

int main(int argc, char* argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);
  std::string url = FLAGS_server_port;
  std::string conf = FLAGS_client_conf;
  std::string test_type = FLAGS_test_type;
  std::string sample_type = FLAGS_sample_type;
  LOG(INFO) << "url = " << url << ";"
            << "client_conf = " << conf << ";"
            << "test_type = " << test_type
            << "sample_type = " << sample_type;
  std::unique_ptr<ServingClient> client;
  if (test_type == "brpc") {
    client.reset(new ServingBrpcClient());
  } else {
    client.reset(new ServingBrpcClient());
  }
  std::vector<std::string> confs;
  confs.push_back(conf);
  if (client->init(confs, url) != 0) {
    LOG(ERROR) << "Failed to init client!";
    return 0;
  }

  PredictorInputs input;
  PredictorOutputs output;
  std::vector<std::string> fetch_name;

  if (sample_type == "fit_a_line") {
    prepare_fit_a_line(input, fetch_name);
  }
  else if (sample_type == "fit_a_line_batch") {
    prepare_fit_a_line(input, fetch_name, 1000);
    batch_size = 1000;
  }
  else if (sample_type == "bert") {
    prepare_bert(input, fetch_name);
  }
  else {
    prepare_fit_a_line(input, fetch_name);
  }

  // if (client->predict(input, output, fetch_name, 0) != 0) {
  //   LOG(ERROR) << "Failed to predict!";
  // }
  // else {
  //   LOG(INFO) << output.print();
  // }
  std::vector<ServingClient*> vec_clients;
  for (int i = 0; i < FLAGS_test_thread_num; ++i) {
    ServingClient* client = new ServingBrpcClient();
      if (client->init(confs, url) != 0) {
      LOG(ERROR) << "Failed to init client!";
      return 0;
    }
    vec_clients.push_back(client);
  }

  Timer total_timeline;
  total_timeline.Start();
  
  std::vector<std::thread> vec_threads;
  for (int i = 0; i < FLAGS_test_thread_num; ++i) {
    vec_threads.push_back(std::thread(thread_func, vec_clients[i], input, fetch_name, i));
  }

  for (std::thread& th : vec_threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  total_timeline.Pause();
  double total_cost = total_timeline.ElapsedMS();
  double each_thread_cost = total_thread_cost / total_thread_count;
  double qps = (double)total_count / (each_thread_cost / 1000.0);

  LOG(INFO) << "\n" 
            << "total cost: " << total_cost << "ms\n"
            << "each thread cost: " << each_thread_cost << "ms\n"
            << "qps: " << qps << "samples/s\n"
            << "total count: " <<  total_count;
  
  return 0;
}
