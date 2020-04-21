/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "general_post.h"

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

using hiai::Engine;
using namespace std;
using namespace ascend::presenter;
using namespace std::__cxx11;

namespace {
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// size of output tensor vector should be 2.
const uint32_t kOutputTensorSize = 2;
const uint32_t kOutputNumIndex = 0;
const uint32_t kOutputTesnorIndex = 1;

const uint32_t kCategoryIndex = 2;
const uint32_t kScorePrecision = 3;

const int32_t kFdFunSuccess = 0;
const int32_t kFdFunFailed = -1;

// bounding box line solid
const uint32_t kLineSolid = 2;

// output image prefix
const string kOutputFilePrefix = "out_";

// boundingbox tensor shape
const static std::vector<uint32_t> kDimDetectionOut = {64, 304, 8};

// num tensor shape
const static std::vector<uint32_t> kDimBBoxCnt = {32};

// port number range
const int32_t kPortMinNumber = 0;
const int32_t kPortMaxNumber = 65535;

// confidence range
const float kConfidenceMin = 0.0;
const float kConfidenceMax = 1.0;

// percent
const int32_t kScorePercent = 100;
//// opencv draw label params.
//const double kFountScale = 0.5;
//const cv::Scalar kFontColor(0, 0, 255);
//const uint32_t kLabelOffset = 11;
//const string kFileSperator = "/";
//
//// opencv color list for boundingbox
//const vector<cv::Scalar> kColors {
//  cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255), cv::Scalar(50, 205, 50),
//  cv::Scalar(139, 85, 26)};
// output tensor index
enum BBoxIndex {kTopLeftX, kTopLeftY, kLowerRigltX, kLowerRightY, kScore};

// IP regular expression
const std::string kIpRegularExpression =
    "^((25[0-5]|2[0-4]\\d|[1]{1}\\d{1}\\d{1}|[1-9]{1}\\d{1}|\\d{1})($|(?!\\.$)\\.)){4}$";

// channel name regular expression
const std::string kChannelNameRegularExpression = "[a-zA-Z0-9/]+";
}
 // namespace

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

GeneralPost::GeneralPost() {
  gd_post_process_config_ = nullptr;
  presenter_channel_ = nullptr;
}

HIAI_StatusT GeneralPost::Init(
  const hiai::AIConfig &config,
  const vector<hiai::AIModelDescription> &model_desc) {

  HIAI_ENGINE_LOG("Begin initialize!");

  //get configuration
  if (gd_post_process_config_ == nullptr) {
    gd_post_process_config_ = std::make_shared<GeneralDetectionPostConfig>();
  }

  // get parameters from graph.config
  for (int index = 0; index < config.items_size(); index++) {
    const ::hiai::AIConfigItem& item = config.items(index);
    const std::string& name = item.name();
    const std::string& value = item.value();
    std::stringstream ss;
    ss << value;
    if (name == "Confidence") {
      ss >> (*gd_post_process_config_).confidence;
      // validate confidence
      if (IsInvalidConfidence(gd_post_process_config_->confidence)) {
        HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE,
                        "Confidence=%s which configured is invalid.",
                        value.c_str());
        return HIAI_ERROR;
      }
    } else if (name == "PresenterIp") {
      // validate presenter server IP
      if (IsInValidIp(value)) {
        HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE,
                        "PresenterIp=%s which configured is invalid.",
                        value.c_str());
        return HIAI_ERROR;
      }
      ss >> (*gd_post_process_config_).presenter_ip;
    } else if (name == "PresenterPort") {
      ss >> (*gd_post_process_config_).presenter_port;
      // validate presenter server port
      if (IsInValidPort(gd_post_process_config_->presenter_port)) {
        HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE,
                        "PresenterPort=%s which configured is invalid.",
                        value.c_str());
        return HIAI_ERROR;
      }
    } else if (name == "ChannelName") {     //channel on presenter server, not camera
      // validate channel name
      if (IsInValidChannelName(value)) {
        HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE,
                        "ChannelName=%s which configured is invalid.",
                        value.c_str());
        return HIAI_ERROR;
      }
      ss >> (*gd_post_process_config_).channel_name;
    }
    // else : nothing need to do
  }

  // call presenter agent, create connection to presenter server
  uint16_t u_port = static_cast<uint16_t>(gd_post_process_config_
      ->presenter_port);
  OpenChannelParam channel_param = { gd_post_process_config_->presenter_ip,
      u_port, gd_post_process_config_->channel_name, ContentType::kVideo };
  Channel *chan = nullptr;
  PresenterErrorCode err_code = OpenChannel(chan, channel_param);
  // open channel failed
  if (err_code != PresenterErrorCode::kNone) {
    HIAI_ENGINE_LOG(HIAI_GRAPH_INIT_FAILED,
                    "Open presenter channel failed, error code=%d", err_code);
    return HIAI_ERROR;
  }

  presenter_channel_.reset(chan);
  HIAI_ENGINE_LOG(HIAI_DEBUG_INFO, "End initialize!");
  return HIAI_OK;
}

bool GeneralPost::IsInValidIp(const std::string &ip) {
  regex re(kIpRegularExpression);
  smatch sm;
  return !regex_match(ip, sm, re);
}

bool GeneralPost::IsInValidPort(int32_t port) {
  return (port <= kPortMinNumber) || (port > kPortMaxNumber);
}

bool GeneralPost::IsInValidChannelName(
    const std::string &channel_name) {
  regex re(kChannelNameRegularExpression);
  smatch sm;
  return !regex_match(channel_name, sm, re);
}

bool GeneralPost::IsInvalidConfidence(float confidence) {
  return (confidence <= kConfidenceMin) || (confidence > kConfidenceMax);
}

//// Process

//bool GeneralPost::IsInvalidResults(float attr, float score,
//                                                const Point &point_lt,
//                                                const Point &point_rb) {
//  // attribute is not face (background)
//  if (std::abs(attr - kAttributeFaceLabelValue) > kAttributeFaceDeviation) {
//    return true;
//  }
//
//  // confidence check
//  if ((score < gd_post_process_config_->confidence)
//      || IsInvalidConfidence(score)) {
//    return true;
//  }
//
//  // rectangle position is a point or not: lt == rb
//  if ((point_lt.x == point_rb.x) && (point_lt.y == point_rb.y)) {
//    return true;
//  }
//  return false;
//}

int32_t GeneralPost::SendImage(uint32_t height, uint32_t width,
                                            uint32_t size, u_int8_t *data, std::vector<DetectionResult>& detection_results) {
  int32_t status = kFdFunSuccess;
  // parameter
  ImageFrame image_frame_para;
  image_frame_para.format = ImageFormat::kJpeg;
  image_frame_para.width = width;
  image_frame_para.height = height;
  image_frame_para.size = size;
  image_frame_para.data = data;
  image_frame_para.detection_results = detection_results;

  PresenterErrorCode p_ret = PresentImage(presenter_channel_.get(),
                                            image_frame_para);
  // send to presenter failed
  if (p_ret != PresenterErrorCode::kNone) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                      "Send JPEG image to presenter failed, error code=%d",
                      p_ret);
    status = kFdFunFailed;
  }

  return status;
}

//HIAI_StatusT GeneralPost::HandleOriginalImage(
//    const std::shared_ptr<EngineTransT> &inference_res) {
//  HIAI_StatusT status = HIAI_OK;
//  std::vector<NewImageParaT> img_vec = inference_res->imgs;
//  // dealing every original image
//  for (uint32_t ind = 0; ind < inference_res->b_info.batch_size; ind++) {
//    uint32_t width = img_vec[ind].img.width;
//    uint32_t height = img_vec[ind].img.height;
//    uint32_t size = img_vec[ind].img.size;
//
//    // call SendImage
//    // 1. call DVPP to change YUV420SP image to JPEG
//    // 2. send image to presenter
//    vector<DetectionResult> detection_results;
//    int32_t ret = SendImage(height, width, size, img_vec[ind].img.data.get(), detection_results);
//    if (ret == kFdFunFailed) {
//      status = HIAI_ERROR;
//      continue;
//    }
//  }
//  return status;
//}


bool GeneralPost::SendSentinel() {
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string",
                        static_pointer_cast<void>(sentinel_msg));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
  }

HIAI_StatusT GeneralPost::FasterRcnnPostProcess(
  const shared_ptr<EngineTrans> &result) {
  vector<Output> outputs = result->inference_res;
  
  if (outputs.size() != kOutputTensorSize) {
    ERROR_LOG("Detection output size does not match.");
    return HIAI_ERROR;
  }

  float *bbox_buffer =
      reinterpret_cast<float *>(outputs[kOutputTesnorIndex].data.get());
  uint32_t *num_buffer =
      reinterpret_cast<uint32_t *>(outputs[kOutputNumIndex].data.get());
  Tensor<uint32_t> tensor_num;
  Tensor<float> tensor_bbox;
  bool ret = true;
  ret = tensor_num.FromArray(num_buffer, kDimBBoxCnt);
  if (!ret) {
    ERROR_LOG("Failed to resolve tensor from array.");
    return HIAI_ERROR;
  }
  ret = tensor_bbox.FromArray(bbox_buffer, kDimDetectionOut);
  if (!ret) {
    ERROR_LOG("Failed to resolve tensor from array.");
    return HIAI_ERROR;
  }

  uint32_t width = result->image_info.width;
  uint32_t height = result->image_info.height;
  uint32_t img_size = result->image_info.size;
  std::vector<DetectionResult> detection_results;

//  vector<BoundingBox> bboxes;
  for (uint32_t attr = 0; attr < result->console_params.output_nums; ++attr) { //iterate over every class
    for (uint32_t bbox_idx = 0; bbox_idx < tensor_num[attr]; ++bbox_idx) {
      uint32_t class_idx = attr * kCategoryIndex;

      //Create Detection result & Point
      DetectionResult one_result;
      Point point_lt, point_rb;
      point_lt.x = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kTopLeftX);
      point_lt.y = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kTopLeftY);
      point_rb.x = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kLowerRigltX);
      point_rb.y = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kLowerRightY);
      one_result.lt = point_lt;
      one_result.rb = point_rb;
//      uint32_t lt_x = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kTopLeftX);
//      uint32_t lt_y = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kTopLeftY);
//      uint32_t rb_x = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kLowerRigltX);
//      uint32_t rb_y = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kLowerRightY);
      float score = tensor_bbox(class_idx, bbox_idx, BBoxIndex::kScore);
      int32_t score_percent =  score * kScorePercent;

      HIAI_ENGINE_LOG(HIAI_DEBUG_INFO,
                      "object=%d score=%f, lt.x=%d, lt.y=%d, rb.x=%d, rb.y=%d", attr, score,
                      point_lt.x, point_lt.y, point_rb.x, point_rb.y);
      stringstream sstream;
      sstream << attr << " ";
      sstream.precision(kScorePrecision);
      sstream << 100 * score << "%";
      string obj_str = sstream.str();
      //push back
      detection_results.emplace_back(one_result);
//      bboxes.push_back( {lt_x, lt_y, rb_x, rb_y, attr, score});
    }
  }
  int32_t ret_send;
  ret_send = SendImage(height, width, img_size, result->image_info.data.get(), detection_results);
  if (ret == kFdFunFailed) {
    return HIAI_ERROR;
  }
//  if (bboxes.empty()) {
//    INFO_LOG("There is none object detected in image %s",
//             result->image_info.path.c_str());
//    return HIAI_OK;
//  }
//
//  cv::Mat mat = cv::imread(result->image_info.path, CV_LOAD_IMAGE_UNCHANGED);
//
//  if (mat.empty()) {
//    ERROR_LOG("Fialed to deal file=%s. Reason: read image failed.",
//              result->image_info.path.c_str());
//    return HIAI_ERROR;
//  }
//  float scale_width = (float)mat.cols / result->image_info.width;
//  float scale_height = (float)mat.rows / result->image_info.height;
//  stringstream sstream;
//  for (int i = 0; i < bboxes.size(); ++i) {
//    cv::Point p1, p2;
//    p1.x = scale_width * bboxes[i].lt_x;
//    p1.y = scale_height * bboxes[i].lt_y;
//    p2.x = scale_width * bboxes[i].rb_x;
//    p2.y = scale_height * bboxes[i].rb_y;
//    cv::rectangle(mat, p1, p2, kColors[i % kColors.size()], kLineSolid);
//
//    sstream.str("");
//    sstream << bboxes[i].attribute << " ";
//    sstream.precision(kScorePrecision);
//    sstream << 100 * bboxes[i].score << "%";
//    string obj_str = sstream.str();
//    cv::putText(mat, obj_str, cv::Point(p1.x, p1.y + kLabelOffset),
//                cv::FONT_HERSHEY_COMPLEX, kFountScale, kFontColor);
//  }
//
//  int pos = result->image_info.path.find_last_of(kFileSperator);
//  string file_name(result->image_info.path.substr(pos + 1));
//  bool save_ret(true);
//  sstream.str("");
//  sstream << result->console_params.output_path << kFileSperator
//          << kOutputFilePrefix << file_name;
//  string output_path = sstream.str();
//  save_ret = cv::imwrite(output_path, mat);
//  if (!save_ret) {
//    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.",
//              result->image_info.path.c_str());
//    return HIAI_ERROR;
//  }
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE) {
HIAI_StatusT ret = HIAI_OK;

// check arg0
if (arg0 == nullptr) {
  ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
  return HIAI_ERROR;
}

// just send to callback function when finished
shared_ptr<EngineTrans> result = static_pointer_cast<EngineTrans>(arg0);
if (result->is_finished) {
  if (SendSentinel()) {
    return HIAI_OK;
  }
  ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
  ERROR_LOG("Please stop this process manually.");
  return HIAI_ERROR;
}

// inference failed
if (result->err_msg.error) {
  ERROR_LOG("%s", result->err_msg.err_msg.c_str());
  return HIAI_ERROR;
}

// arrange result
  return FasterRcnnPostProcess(result);
}
