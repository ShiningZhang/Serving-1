function(PROTOBUF_GENERATE_SERVING_CPP FOR_STUB SRCS HDRS )
  if(NOT ARGN)
      message(SEND_ERROR "Error: PROTOBUF_GENERATE_SERVING_CPP() called without any proto files")
          return()
	    endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
      foreach(DIR ${Protobuf_IMPORT_DIRS})
        get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
        list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
        if(${_contains_already} EQUAL -1)
            list(APPEND _protobuf_include_path -I ${ABS_PATH})
        endif()
      endforeach()
    endif()

    set(${SRCS})
    set(${HDRS})
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(FIL_WE ${FIL} NAME_WE)

      list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
      list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

      set(PDCODEGEN "${CMAKE_BINARY_DIR}/core/pdcodegen/pdcodegen")
			if(NOT ${FOR_STUB})
          add_custom_command(
              OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
                     "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
              COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
                  ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                  --proto_path=${CMAKE_SOURCE_DIR}/core/predictor/proto
                  ${_protobuf_include_path} ${ABS_FIL}
              DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE} ${PDCODEGEN}
              COMMENT "Running Paddle-serving C++ protocol buffer compiler on ${FIL}"
              VERBATIM)
			else()
					add_custom_command(
							OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.cc"
										 "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.h"
							COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
									ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
        					--plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
									--proto_path=${CMAKE_CURRENT_BINARY_DIR}
									${_protobuf_include_path} ${ABS_FIL}
							DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE} ${GRPC_CPP_PLUGIN}
							COMMENT "Running Paddle-serving C++ protocol buffer grpc on ${FIL}"
							VERBATIM)
			endif()
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto"
    COMMAND sed 
        ARGS "s/option cc_generic_services = true/option cc_generic_services = false/g"
        ${CMAKE_SOURCE_DIR}/core/sdk-cpp/proto/general_model_service.proto
        > ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto
    COMMENT "Running Paddle-serving C++ proto grpc"
    VERBATIM)

# 添加自定义target
add_custom_target(generate_message ALL
DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto
COMMENT "generate message target"
VERBATIM
)

list(APPEND protos ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto)
PROTOBUF_GENERATE_SERVING_CPP(TRUE PROTO_SRCS PROTO_HDRS ${protos})
LIST(APPEND grpc_stub_srcs ${PROTO_SRCS})
list(APPEND protos ${CMAKE_SOURCE_DIR}/core//predictor/proto/pds_option.proto
        ${CMAKE_SOURCE_DIR}/core//predictor/proto/builtin_format.proto)
PROTOBUF_GENERATE_SERVING_CPP(FALSE PROTO_SRCS PROTO_HDRS ${protos})
LIST(APPEND grpc_stub_srcs ${PROTO_SRCS})

#Generate grpc stub of general_model_service
# add_custom_command(
#     OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/general_model_service.grpc.pb.cc"
#            "${CMAKE_CURRENT_BINARY_DIR}/general_model_service.grpc.pb.h"
#     COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
#         ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
#         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
#         --proto_path=${CMAKE_CURRENT_BINARY_DIR}
#         -I ${CMAKE_SOURCE_DIR}/core//predictor/proto
#         ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto
#     DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${GRPC_CPP_PLUGIN}
#             ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.proto
#     COMMENT "Running Paddle-serving C++ protocol buffer compiler on general_model_service.grpc"
#     VERBATIM)
# LIST(APPEND grpc_stub_srcs ${CMAKE_CURRENT_BINARY_DIR}/general_model_service.pb.cc)
add_library(grpc_stub ${grpc_stub_srcs})
add_dependencies(grpc_stub configure protobuf grpc++_unsecure)
target_link_libraries(grpc_stub grpc++_unsecure configure protobuf)
