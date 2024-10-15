#!/usr/bin/python3

# Before run this code, 
# Execute '/usr/local/cuda/bin/nvcc -Xcompiler -fPIC -shared -o concurrent_model.so concurrent_model.cu' first.

from concurrent.futures import process
from urllib.parse import uses_query
import time, os
import sys
from math import factorial, exp
import multiprocessing
from multiprocessing import Process, Queue, Value
import numpy as np
import statistics as st
from numpy import random
import ctypes
from ctypes import *
from threading import Thread

def alexnet_inference(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list):
    pid = os.getpid()
    print('Alexnet Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./concurrent_model.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_alexnet = dll.host2gpu_alexnet
    host2gpu_alexnet.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                      POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    inference_alexnet = dll.inference_alexnet
    inference_alexnet.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                      POINTER(c_float),POINTER(c_float),POINTER(c_float)] 

    free_alexnet = dll.free_alexnet
    free_alexnet.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float)] 
              
    A_L1_N = POINTER(c_float)()
    A_L2_N = POINTER(c_float)()
    A_L3_N = POINTER(c_float)()
    A_L4_N = POINTER(c_float)()
    A_L5_N = POINTER(c_float)()
    A_L6_N = POINTER(c_float)()
    A_L7_N = POINTER(c_float)()
    A_L8_N = POINTER(c_float)()

    A_L1_b = POINTER(c_float)()
    A_L2_b = POINTER(c_float)()
    A_L3_b = POINTER(c_float)()
    A_L4_b = POINTER(c_float)()
    A_L5_b = POINTER(c_float)()
    A_L6_b = POINTER(c_float)()
    A_L7_b = POINTER(c_float)()
    A_L8_b = POINTER(c_float)()

    A_L1_w = POINTER(c_float)()
    A_L2_w = POINTER(c_float)()
    A_L3_w = POINTER(c_float)()
    A_L4_w = POINTER(c_float)()
    A_L5_w = POINTER(c_float)()
    A_L6_w = POINTER(c_float)()
    A_L7_w = POINTER(c_float)()
    A_L8_w = POINTER(c_float)()

    A_L1_pool = POINTER(c_float)()
    A_L2_pool = POINTER(c_float)()
    A_L5_pool = POINTER(c_float)()
    A_L1_norm = POINTER(c_float)()
    A_L2_norm = POINTER(c_float)()
    A_Result_N = POINTER(c_float)()

    host2gpu_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N)

    #warm up
    for i in range(5):
        inference_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                        A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                        A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                        A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                        A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                        A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                        A_L1_pool,A_L2_pool,A_L5_pool,
                        A_L1_norm,A_L2_norm,A_Result_N)

    stopping.value = stopping.value + 1
    
    print("Alexnet Ready!")
    while(stopping.value < 3):
        continue

    starttime = time.time()
    starttime_list.append(starttime)
    processed_requests = 0
    while True:
        if(user_request.empty()):
            if(producer_status.value == 0):        #user_request는 비어있지만 request creator는 아직 작동중
                continue
            elif(producer_status.value == 1):      #user_request는 비어있고 request creator 작동 끝 --> break
                break
        elif(user_request.qsize() > 0):
            file = open("results/concurrent_results.txt","a")
            request_count = count.qsize()       #처리된 request가 X개를 넘으면 inference 종료
            if request_count > (total - 1):
                print("PID: {0}, Alexnet inference process finish, {1} requests processed".format(pid, processed_requests))
                file.close()
                finisher.value = 1 
                break
            elif(random_model[0] == 'alexnet'):         
                queueing_delay = time.time()-user_request.get()
                if queueing_delay < 0:
                    queueing_delay = 0
                del random_model[0]
                #print(f"Queueing delay : {(queueing_delay):.6f} seconds")
                start_time = time.time()
                inference_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                                A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                                A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                                A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                                A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                                A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                                A_L1_pool,A_L2_pool,A_L5_pool,
                                A_L1_norm,A_L2_norm,A_Result_N)
                end_time = time.time()
                processed_requests += 1
                pure_inference_time = end_time-start_time
                sum_time = queueing_delay + pure_inference_time
                file.write("Alexnet" f"\t{(queueing_delay):.6f} \t {(pure_inference_time):.6f} \t {(sum_time):.6f}\n") 
                count.put(0)
        file.close()
    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Alexnet inference time : {(endtime-starttime):.6f} seconds")

    free_alexnet(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N)

def resnet18_inference(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list):
    pid = os.getpid()
    print('Resnet18 Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./concurrent_model.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_resnet18 = dll.host2gpu_resnet18
    host2gpu_resnet18.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    inference_resnet18 = dll.inference_resnet18
    inference_resnet18.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    free_resnet18 = dll.free_resnet18
    free_resnet18.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    R_L1_N = POINTER(c_float)()
    R_L2_N = POINTER(c_float)()
    R_L3_N = POINTER(c_float)()
    R_L4_N = POINTER(c_float)()
    R_L5_N = POINTER(c_float)()
    R_L6_N = POINTER(c_float)()
    R_L7_N = POINTER(c_float)()
    R_L8_N = POINTER(c_float)()
    R_L9_N = POINTER(c_float)()
    R_L10_N = POINTER(c_float)()
    R_L11_N = POINTER(c_float)()
    R_L12_N = POINTER(c_float)()
    R_L13_N = POINTER(c_float)()
    R_L14_N = POINTER(c_float)()
    R_L15_N = POINTER(c_float)()
    R_L16_N = POINTER(c_float)()
    R_L17_N = POINTER(c_float)()
    R_L18_N = POINTER(c_float)()

    R_L1_w = POINTER(c_float)()
    R_L2_w = POINTER(c_float)()
    R_L3_w = POINTER(c_float)()
    R_L4_w = POINTER(c_float)()
    R_L5_w = POINTER(c_float)()
    R_L6_w = POINTER(c_float)()
    R_L7_w = POINTER(c_float)()
    R_L8_w = POINTER(c_float)()
    R_L9_w = POINTER(c_float)()
    R_L10_w = POINTER(c_float)()
    R_L11_w = POINTER(c_float)()
    R_L12_w = POINTER(c_float)()
    R_L13_w = POINTER(c_float)()
    R_L14_w = POINTER(c_float)()
    R_L15_w = POINTER(c_float)()
    R_L16_w = POINTER(c_float)()
    R_L17_w = POINTER(c_float)()
    R_B3_w = POINTER(c_float)()
    R_B4_w = POINTER(c_float)()
    R_B5_w = POINTER(c_float)()

    R_L1_G = POINTER(c_float)()
    R_L2_G = POINTER(c_float)()
    R_L3_G = POINTER(c_float)()
    R_L4_G = POINTER(c_float)()
    R_L5_G = POINTER(c_float)()
    R_L6_G = POINTER(c_float)()
    R_L7_G = POINTER(c_float)()
    R_L8_G = POINTER(c_float)()
    R_L9_G = POINTER(c_float)()
    R_L10_G = POINTER(c_float)()
    R_L11_G = POINTER(c_float)()
    R_L12_G = POINTER(c_float)()
    R_L13_G = POINTER(c_float)()
    R_L14_G = POINTER(c_float)()
    R_L15_G = POINTER(c_float)()
    R_L16_G = POINTER(c_float)()
    R_L17_G = POINTER(c_float)()
    R_B3_G = POINTER(c_float)()
    R_B4_G = POINTER(c_float)()
    R_B5_G = POINTER(c_float)()

    R_L1_B = POINTER(c_float)()
    R_L2_B = POINTER(c_float)()
    R_L3_B = POINTER(c_float)()
    R_L4_B = POINTER(c_float)()
    R_L5_B = POINTER(c_float)()
    R_L6_B = POINTER(c_float)()
    R_L7_B = POINTER(c_float)()
    R_L8_B = POINTER(c_float)()
    R_L9_B = POINTER(c_float)()
    R_L10_B = POINTER(c_float)()
    R_L11_B = POINTER(c_float)()
    R_L12_B = POINTER(c_float)()
    R_L13_B = POINTER(c_float)()
    R_L14_B = POINTER(c_float)()
    R_L15_B = POINTER(c_float)()
    R_L16_B = POINTER(c_float)()
    R_L17_B = POINTER(c_float)()
    R_B3_B = POINTER(c_float)()
    R_B4_B = POINTER(c_float)()
    R_B5_B = POINTER(c_float)()

    R_L1_M = POINTER(c_float)()
    R_L2_M = POINTER(c_float)()
    R_L3_M = POINTER(c_float)()
    R_L4_M = POINTER(c_float)()
    R_L5_M = POINTER(c_float)()
    R_L6_M = POINTER(c_float)()
    R_L7_M = POINTER(c_float)()
    R_L8_M = POINTER(c_float)()
    R_L9_M = POINTER(c_float)()
    R_L10_M = POINTER(c_float)()
    R_L11_M = POINTER(c_float)()
    R_L12_M = POINTER(c_float)()
    R_L13_M = POINTER(c_float)()
    R_L14_M = POINTER(c_float)()
    R_L15_M = POINTER(c_float)()
    R_L16_M = POINTER(c_float)()
    R_L17_M = POINTER(c_float)()
    R_B3_M = POINTER(c_float)()
    R_B4_M = POINTER(c_float)()
    R_B5_M = POINTER(c_float)()

    R_L1_V = POINTER(c_float)()
    R_L2_V = POINTER(c_float)()
    R_L3_V = POINTER(c_float)()
    R_L4_V = POINTER(c_float)()
    R_L5_V = POINTER(c_float)()
    R_L6_V = POINTER(c_float)()
    R_L7_V = POINTER(c_float)()
    R_L8_V = POINTER(c_float)()
    R_L9_V = POINTER(c_float)()
    R_L10_V = POINTER(c_float)()
    R_L11_V = POINTER(c_float)()
    R_L12_V = POINTER(c_float)()
    R_L13_V = POINTER(c_float)()
    R_L14_V = POINTER(c_float)()
    R_L15_V = POINTER(c_float)()
    R_L16_V = POINTER(c_float)()
    R_L17_V = POINTER(c_float)()
    R_B3_V = POINTER(c_float)()
    R_B4_V = POINTER(c_float)()
    R_B5_V = POINTER(c_float)()

    R_FC_b = POINTER(c_float)()
    R_FC_w = POINTER(c_float)()

    R_L3_basic = POINTER(c_float)()
    R_L5_basic = POINTER(c_float)()
    R_L7_basic = POINTER(c_float)()
    R_L9_basic = POINTER(c_float)()
    R_L11_basic = POINTER(c_float)()
    R_L13_basic = POINTER(c_float)()
    R_L15_basic = POINTER(c_float)()
    R_L17_basic = POINTER(c_float)()
    R_B3_basic = POINTER(c_float)()
    R_B4_basic = POINTER(c_float)()
    R_B5_basic = POINTER(c_float)()
    R_L1_bn = POINTER(c_float)()
    R_L2_bn = POINTER(c_float)()
    R_L3_bn = POINTER(c_float)()
    R_L4_bn = POINTER(c_float)()
    R_L5_bn = POINTER(c_float)()
    R_L6_bn = POINTER(c_float)()
    R_L7_bn = POINTER(c_float)()
    R_L8_bn = POINTER(c_float)()
    R_L9_bn = POINTER(c_float)()
    R_L10_bn = POINTER(c_float)()
    R_L11_bn = POINTER(c_float)()
    R_L12_bn = POINTER(c_float)()
    R_L13_bn = POINTER(c_float)()
    R_L14_bn = POINTER(c_float)()
    R_L15_bn = POINTER(c_float)()
    R_L16_bn = POINTER(c_float)()
    R_L17_bn = POINTER(c_float)()
    R_B3_bn = POINTER(c_float)()
    R_B4_bn = POINTER(c_float)()
    R_B5_bn = POINTER(c_float)()
    R_L1_pool = POINTER(c_float)()
    R_FC_N = POINTER(c_float)()
    R_Result_N = POINTER(c_float)()

    host2gpu_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

    #warm up
    for i in range(5):
        inference_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                            R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                            R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                            R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                            R_L17_N,R_L18_N,
                            R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                            R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                            R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                            R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                            R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                            R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                            R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                            R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                            R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                            R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                            R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                            R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                            R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                            R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                            R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                            R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                            R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                            R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                            R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                            R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                            R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                            R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                            R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                            R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                            R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                            R_FC_b,R_FC_w,
                            R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                            R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                            R_B3_basic,R_B4_basic,R_B5_basic,
                            R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                            R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                            R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                            R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                            R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                            R_L1_pool,R_FC_N,R_Result_N) 

    stopping.value = stopping.value + 1

    print("Resnet18 Ready!")
    while(stopping.value < 3):
        continue

    starttime = time.time()
    starttime_list.append(starttime)
    processed_requests = 0
    while True:
        if(user_request.empty()):
            if(producer_status.value == 0):        #user_request는 비어있지만 request creator는 아직 작동중
                continue
            elif (producer_status.value == 1):      #user_request는 비어있고 request creator 작동 끝 --> break
                break
        elif(user_request.qsize() > 0):
            file = open("results/concurrent_results.txt","a")
            request_count = count.qsize()       #처리된 request가 X개를 넘으면 inference 종료
            if request_count > (total - 1):
                print("PID: {0}, Resnet18 inference process finish, {1} requests processed".format(pid, processed_requests))
                file.close()
                finisher.value = 1 
                break 
            elif(random_model[0] == 'resnet18'): 
                queueing_delay = time.time()-user_request.get()
                if queueing_delay < 0:
                    queueing_delay = 0
                del random_model[0]
                #print(f"Queueing delay : {(queueing_delay):.6f} seconds")
                start_time = time.time()
                inference_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                                    R_L17_N,R_L18_N,
                                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                                    R_FC_b,R_FC_w,
                                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                                    R_B3_basic,R_B4_basic,R_B5_basic,
                                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                                    R_L1_pool,R_FC_N,R_Result_N)
                end_time = time.time()
                processed_requests += 1
                pure_inference_time = end_time-start_time    
                sum_time = queueing_delay + pure_inference_time
                file.write("Resnet" f"\t{(queueing_delay):.6f} \t {(pure_inference_time):.6f} \t {(sum_time):.6f}\n") 
                count.put(0)
        file.close()
    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Resnet18 inference time : {(endtime-starttime):.6f} seconds")
        
    free_resnet18(R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

def vgg16_inference(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list):
    pid = os.getpid()
    print('Vgg16 Inference PID : {0}'.format(pid))

    dll = ctypes.CDLL('./concurrent_model.so', mode=ctypes.RTLD_GLOBAL)
    host2gpu_vgg16 = dll.host2gpu_vgg16
    host2gpu_vgg16.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    inference_vgg16 = dll.inference_vgg16
    inference_vgg16.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float)] 

    free_vgg16 = dll.free_vgg16
    free_vgg16.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float)] 

    V_L1_N = POINTER(c_float)()
    V_L2_N = POINTER(c_float)()
    V_L3_N = POINTER(c_float)()
    V_L4_N = POINTER(c_float)()
    V_L5_N = POINTER(c_float)()
    V_L6_N = POINTER(c_float)()
    V_L7_N = POINTER(c_float)()
    V_L8_N = POINTER(c_float)()
    V_L9_N = POINTER(c_float)()
    V_L10_N = POINTER(c_float)()
    V_L11_N = POINTER(c_float)()
    V_L12_N = POINTER(c_float)()
    V_L13_N = POINTER(c_float)()
    V_L14_N = POINTER(c_float)()
    V_L15_N = POINTER(c_float)()
    V_L16_N = POINTER(c_float)()

    V_L1_b = POINTER(c_float)()
    V_L2_b = POINTER(c_float)()
    V_L3_b = POINTER(c_float)()
    V_L4_b = POINTER(c_float)()
    V_L5_b = POINTER(c_float)()
    V_L6_b = POINTER(c_float)()
    V_L7_b = POINTER(c_float)()
    V_L8_b = POINTER(c_float)()
    V_L9_b = POINTER(c_float)()
    V_L10_b = POINTER(c_float)()
    V_L11_b = POINTER(c_float)()
    V_L12_b = POINTER(c_float)()
    V_L13_b = POINTER(c_float)()
    V_L14_b = POINTER(c_float)()
    V_L15_b = POINTER(c_float)()
    V_L16_b = POINTER(c_float)()

    V_L1_w = POINTER(c_float)()
    V_L2_w = POINTER(c_float)()
    V_L3_w = POINTER(c_float)()
    V_L4_w = POINTER(c_float)()
    V_L5_w = POINTER(c_float)()
    V_L6_w = POINTER(c_float)()
    V_L7_w = POINTER(c_float)()
    V_L8_w = POINTER(c_float)()
    V_L9_w = POINTER(c_float)()
    V_L10_w = POINTER(c_float)()
    V_L11_w = POINTER(c_float)()
    V_L12_w = POINTER(c_float)()
    V_L13_w = POINTER(c_float)()
    V_L14_w = POINTER(c_float)()
    V_L15_w = POINTER(c_float)()
    V_L16_w = POINTER(c_float)()

    V_L2_pool = POINTER(c_float)()
    V_L4_pool = POINTER(c_float)()
    V_L7_pool = POINTER(c_float)()
    V_L10_pool = POINTER(c_float)()
    V_L13_pool = POINTER(c_float)()
    V_Result_N = POINTER(c_float)()

    host2gpu_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                    V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                    V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                    V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                    V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                    V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                    V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                    V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                    V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                    V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                    V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                    V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                    V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                    V_L13_pool,V_Result_N)

    #warm up
    for i in range(5):
        inference_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                        V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                        V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                        V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                        V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                        V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                        V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                        V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                        V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                        V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                        V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                        V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                        V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                        V_L13_pool,V_Result_N)

    stopping.value = stopping.value + 1
    
    print("Vgg16 Ready!")
    while(stopping.value < 3):
        continue

    starttime = time.time()
    starttime_list.append(starttime)
    processed_requests = 0
    while True:
        if(user_request.empty()):
            if(producer_status.value == 0):        #user_request는 비어있지만 request creator는 아직 작동중
                continue
            elif (producer_status.value == 1):      #user_request는 비어있고 request creator 작동 끝 --> break
                break
        elif(user_request.qsize() > 0):
            file = open("concurrent_result.txt","a")
            request_count = count.qsize()       #처리된 request가 X개를 넘으면 inference 종료
            if request_count > (total - 1):
                print("PID: {0}, Vgg16 inference process finish, {1} requests processed".format(pid, processed_requests))
                file.close()
                finisher.value = 1 
                break 
            elif(random_model[0] == 'vgg16'):                     
                queueing_delay = time.time()-user_request.get()
                if queueing_delay < 0:
                    queueing_delay = 0
                del random_model[0]
                #print(f"Queueing delay : {(queueing_delay):.6f} seconds")
                start_time = time.time()
                inference_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                                V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                                V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                                V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                                V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                                V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                                V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                                V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                                V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                                V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                                V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                                V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                                V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                                V_L13_pool,V_Result_N)
                end_time = time.time()
                processed_requests += 1
                pure_inference_time = end_time-start_time 
                sum_time = queueing_delay + pure_inference_time
                file.write("Vgg16" f"\t{(queueing_delay):.6f} \t {(pure_inference_time):.6f} \t {(sum_time):.6f}\n") 
                count.put(0)
        file.close()
    endtime = time.time()
    endtime_list.append(endtime)
    print(f"total Vgg16 inference time : {(endtime-starttime):.6f} seconds")

    free_vgg16(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                V_L13_pool,V_Result_N)

def request_arrival(user_request, queue_size, arrival_time, stopping, producer_status, finisher):
    pid = os.getpid()
    print('request producer PID: ',pid)

    # count1 = 0
    # count2 = 0
    # count3 = 0
    not_processed = 0
    poisson_list = []   

    time.sleep(2)  

    while(stopping.value < 3):
        continue

    print("request creator start!")
    not_processed = 0

    _lambda = arrival_time
    mod_lambda = _lambda * 1000000                 

    poisson_list = random.poisson(mod_lambda, 10000)       
    poisson_list = poisson_list / 1000000 

    for timeslot in poisson_list:
        time.sleep (timeslot)
        requested_time = time.time()
        if (user_request.qsize() <= queue_size and finisher.value != 1):                        
            user_request.put(requested_time)
        elif(finisher.value == 1):
            break                                           # inference 1000개 끝나면 request creater도 종료
        else:
            not_processed += 1
            continue 
    producer_status.value = 1
    # print("Request models: Alex: {}, Res: {}, Vgg: {}\n".format(count1,count2,count3))
    print(pid, ": request producer finish")
    print('{0} requests not added to request queue'.format(not_processed))

stopping = Value('i', 0)  #for simultaneous starting of inferences between processes
producer_status = Value('i', 0)     #if request producer is done, turn into 1
finisher = Value('i', 0)     #if request producer is done, turn into 1

if __name__ == '__main__':

    user_request = Queue()
    count = Queue()

    manager = multiprocessing.Manager()
    starttime_list = manager.list()
    endtime_list = manager.list()
    random_model = manager.list()
    with open("common/queue_model.txt") as f:
        for line in f:
            random_model.append(line.strip()) 

    total = 1000
    avg_inference_time = 0.076218889
    arrival_time = avg_inference_time * 100 / 95
     
    rho = avg_inference_time/arrival_time
    queue_size = round((rho*rho)/(1-rho))

    children = []

    start_time = time.time()

    children.append(Thread(target=resnet18_inference, args=(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list)))
    children.append(Thread(target=alexnet_inference, args=(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list)))
    children.append(Thread(target=vgg16_inference, args=(count, user_request, random_model, total, stopping, producer_status, finisher, starttime_list, endtime_list)))

    p0 = Thread(target = request_arrival, args = (user_request, queue_size, arrival_time, stopping, producer_status, finisher))    

    children.append(p0)             #request generator into children list

    for child in children:
        child.start()

    for child in children:
        child.join()

    starttime = min(starttime_list)
    endtime = max(endtime_list)
    total_inference_time = endtime-starttime
    end_time = time.time()
    total_time = end_time-start_time

    print(queue_size)
    print(f"total inferene time : {(total_inference_time):.6f}")
    print(f"total time : {(total_time):.6f}")
    print('total processed requests : {0}'.format(count.qsize()))







