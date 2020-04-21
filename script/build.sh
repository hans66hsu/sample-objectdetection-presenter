#!/bin/bash
#
#   =======================================================================
#
# Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   1 Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   2 Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   3 Neither the names of the copyright holders nor the names of the
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#   =======================================================================

app_type="objectdetection"

script_path="$( cd "$(dirname "$0")" ; pwd -P )"

app_path="${script_path}/.."
main()
{
    # check DDK_HOME
    if [ ! -n ${DDK_HOME} ];then
        echo "Can not find DDK_HOME env, please set it in environment!."
        exit 1
    fi

    echo "Clear app build path..."
    rm -rf ${app_path}/objectdetectionapp/out

    echo "Build main..."
    make -C ${app_path}/objectdetectionapp 1>/dev/null
    if [ $? -ne 0 ];then
        exit 1
    fi

    for file in `find ${app_path}/objectdetectionapp -name "Makefile"`
    do
        if [ ${file} == "${app_path}/objectdetectionapp/Makefile" ];then
            continue
        fi
        path=`dirname ${file}`
        lib_path_name=`basename ${path}`
        echo "Build ${lib_path_name} lib..."
        make clean -C ${path} 1>/dev/null
        if [ $? -ne 0 ];then
            exit 1
        fi
        make install -C ${path} 1>/dev/null
        if [ $? -ne 0 ];then
            exit 1
        fi
    done
    
#    cp ${app_path}/run_object_detection_faster_rcnn.py ${app_path}/objectdetection/out
#    cp ${app_path}/objectdetection/graph.template ${app_path}/objectdetection/out

    echo "Finish to Build app."
    exit 0
}

main
