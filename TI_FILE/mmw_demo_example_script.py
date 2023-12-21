# ****************************************************************************
# * (C) Copyright 2020, Texas Instruments Incorporated. - www.ti.com
# ****************************************************************************
# *
# *  Redistribution and use in source and binary forms, with or without
# *  modification, are permitted provided that the following conditions are
# *  met:
# *
# *    Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# *    Redistributions in binary form must reproduce the above copyright
# *    notice, this list of conditions and the following disclaimer in the
# *     documentation and/or other materials provided with the distribution.
# *
# *    Neither the name of Texas Instruments Incorporated nor the names of its
# *    contributors may be used to endorse or promote products derived from
# *    this software without specific prior written permission.
# *
# *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# *  PARTICULAR TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# *  A PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  OWNER OR
# *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# *  EXEMPLARY, ORCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# *  LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY, OR TORT (INCLUDING
# *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# ****************************************************************************


# ****************************************************************************
# Sample mmW demo UART output parser script - should be invoked using python3
#       ex: python3 mmw_demo_example_script.py <recorded_dat_file_from_Visualizer>.dat
#
# Notes:
#   1. The parser_mmw_demo script will output the text version 
#      of the captured files on stdio. User can redirect that output to a log file, if desired
#   2. This example script also outputs the detected point cloud data in mmw_demo_output.csv 
#      to showcase how to use the output of parser_one_mmw_demo_output_packet
# ****************************************************************************

import os
import sys
# import the parser function 
from TI_FILE.parser_mmw_demo import parser_one_mmw_demo_output_packet

##################################################################################
# INPUT CONFIGURATION
##################################################################################
# get the captured file name (obtained from Visualizer via 'Record Start')
# if (len(sys.argv) > 1):
#     capturedFileName=sys.argv[1]
# else:
#     print ("Error: provide file name of the saved stream from Visualizer for OOB demo")
#     exit()

def Analytical_data(data_path, index, label, save_path):
    """
    文件转化，将dat文件转化为csv文件
    :param data_path: 文件路径
    :param index: 文件序号
    :param label: 文件标签
    :param save_path: 保存路径
    :return:
    """
    capturedFileName=data_path

    ##################################################################################
    # USE parser_mmw_demo SCRIPT TO PARSE ABOVE INPUT FILES
    ##################################################################################

    # Read the entire file
    fp = open(capturedFileName, 'rb')
    readNumBytes = os.path.getsize(capturedFileName)
    print("readNumBytes: ", readNumBytes)
    allBinData = fp.read()  # 默认读取全部的文件
    print("allBinData: ", allBinData[0], allBinData[1], allBinData[2], allBinData[3])
    print("allBinDataSize: ", sys.getsizeof(allBinData))  # 可以看到和实际大小似乎不一致 后续需要看是否需要解决
    fp.close()

    # init local variables
    totalBytesParsed = 0  # 解析的全部数据
    numFramesParsed = 0  # 解析的帧数
    coord=[]

    # parser_one_mmw_demo_output_packet extracts only one complete frame at a time  一次只读取一个完整的帧
    # so call this in a loop till end of file  循环获得文件的全部内容
    while totalBytesParsed < readNumBytes:

        # parser_one_mmw_demo_output_packet function already prints the
        # parsed data to stdio. So showcasing only saving the data to arrays
        # here for further custom processing
        # 解析结果 首部开始地址 总数据包字节数 检测到的个体数 TLV数量 子帧号
        parser_result, \
        headerStartIndex, \
        totalPacketNumBytes, \
        numDetObj, \
        numTlv, \
        subFrameNumber, \
        detectedX_array, \
        detectedY_array, \
        detectedZ_array, \
        detectedV_array, \
        detectedRange_array, \
        detectedAzimuth_array, \
        detectedElevation_array, \
        detectedSNR_array, \
        detectedNoise_array = parser_one_mmw_demo_output_packet(allBinData[totalBytesParsed::1], readNumBytes-totalBytesParsed)

        # Check the parser result
        print ("Parser result: ", parser_result)
        if parser_result == 0:
            totalBytesParsed += (headerStartIndex+totalPacketNumBytes)
            numFramesParsed+=1
            print("totalBytesParsed: ", totalBytesParsed)
            ##################################################################################
            # TODO: use the arrays returned by above parser as needed.
            # For array dimensions, see help(parser_one_mmw_demo_output_packet)
            # help(parser_one_mmw_demo_output_packet)
            ##################################################################################

            #  For example, dump all S/W objects to a csv file
            import csv
            if numFramesParsed == 1:
                democsvfile = open(f'{save_path}/{index}_{label}.csv', 'w', newline='')
                demoOutputWriter = csv.writer(democsvfile, delimiter=',',quotechar='', quoting=csv.QUOTE_NONE)
                demoOutputWriter.writerow(["frame","DetObj","x","y","z","v","distance","azimuth","elevation","snr","noise"])

            for obj in range(numDetObj):
                demoOutputWriter.writerow([numFramesParsed-1, obj, detectedX_array[obj], \
                                        detectedY_array[obj], \
                                        detectedZ_array[obj], \
                                        detectedV_array[obj], \
                                        detectedRange_array[obj], \
                                        detectedAzimuth_array[obj], \
                                        detectedElevation_array[obj], \
                                        detectedSNR_array[obj], \
                                        detectedNoise_array[obj]])

        else:
            # error in parsing; exit the loop
            break

    # All processing done; Exit
    print("numFramesParsed: ", numFramesParsed)

