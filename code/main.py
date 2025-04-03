import cv2
import uvicorn
import numpy as np
import time
from picture_process import zhifangtu
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import multiprocessing
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response


from fastapi.staticfiles import StaticFiles

from fastapi.responses import StreamingResponse, FileResponse
from ultralytics import YOLO
import asyncio

import queue
import io
from fastapi import UploadFile, File, HTTPException



camera_width = 640
camera_height = 480
garbage_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

h_lower = 100
h_upper = 200
s_lower = 100
s_upper = 200
v_lower = 100
v_upper = 200


class OptimizedTracker:
    def __init__(self, model_path):
        # 初始化YOLO模型
        self.model = YOLO(model_path, task='detect')

        # 白色HSV范围
        self.white_lower = np.array([127, 0, 183])
        self.white_upper = np.array([179, 227, 255])

        # 预分配内存
        self.kernel = np.ones((5, 5), np.uint8)

    def process_white_region(self, frame, roi):
        """快速处理白色区域"""
        # 提取ROI
        frame = zhifangtu(frame)
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # HSV转换和处理
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        # cv2.imshow("Mask After Morph Close", mask)
        # 单次腐蚀，去除噪点
        mask = cv2.erode(mask, self.kernel, iterations=1)
        # 膨胀操作
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        # 高斯滤波
        closing = cv2.GaussianBlur(mask, (5, 5), 0)
        # 边缘检测
        mask = cv2.Canny(closing, 10, 20)
        # cv2.imshow("process", mask)
        return mask

    def find_white_quadrilateral(self, mask):
        """查找白色四边形"""
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, 0

        # 获取最大轮廓
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)

        # 计算中心点
        M = cv2.moments(max_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = None

        return max_cnt, center, area

# 用于在子进程中执行推理的函数
def inference_worker(queue_in, queue_out, user_queue_in, user_queue_out, model_path):
    tracker = OptimizedTracker(model_path)
    # 获取相机参数
    width = camera_width
    height = camera_height
    camera_center = (width // 2, height // 2)

    while not stop_event_2.is_set():
        is_user_image = False
        try:
            if not user_queue_in.empty():
                frame = user_queue_in.get()
                is_user_image = True
                print("user_image", frame.shape)
            else:
                frame = queue_in.get()
                if frame is None:  # 用于退出的信号
                    break

            if (not queue_out.full()) or is_user_image:
                results = tracker.model(frame, verbose=False)
                car_area = None
                car_offset = None
                white_area = None
                white_offset = None

                
                if results[0].boxes:
                    # 获取小车框
                    frame = results[0].plot()
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    car_area = (x2 - x1) * (y2 - y1)

                    # 计算小车中心
                    car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    car_offset = (car_center[0] - camera_center[0],
                                camera_center[1] - car_center[1])
                    # 绘制小车中心点
                    cv2.circle(frame, car_center, 5, (0, 0, 255), -1)  # 绘制红色圆点表示小车中心

                    # 绘制摄像头中心点
                    cv2.circle(frame, camera_center, 5,
                            (255, 0, 0), -1)  # 绘制蓝色圆点表示摄像头中心
                    
                    if not is_user_image:
                        # 在图像上显示文本
                        cv2.putText(frame, f'Car Area: {car_area}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 处理白色区域
                    roi = (x1, y1, x2-x1, y2-y1)

                    white_mask = tracker.process_white_region(frame, roi)

                    # 查找白色四边形
                    contour, white_center, white_area = tracker.find_white_quadrilateral(
                        white_mask)
                    

                    if white_center:

                        # 调整白色区域中心点坐标（相对于整个图像）
                        white_center = (white_center[0] + x1, white_center[1] + y1)
                        white_offset = (white_center[0] - camera_center[0],
                                        camera_center[1] - white_center[1])

                        cv2.circle(frame, white_center, 5,
                                (45, 184, 255), -1)  # 绘制红色圆点表示小车中心

                        # 在图像上绘制结果
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if contour is not None:
                            contour_shifted = contour + np.array([x1, y1])
                            cv2.drawContours(
                                frame, [contour_shifted], -1, (0, 0, 255), 2)
                            
                if is_user_image:
                    user_queue_out.put({
                    'frame': frame,
                    'car_area': car_area,
                    'car_offset': car_offset,
                    'white_area': white_area,
                    'white_offset': white_offset
                })
                else:
                    queue_out.put({
                        'frame': frame,
                        'car_area': car_area,
                        'car_offset': car_offset,
                        'white_area': white_area,
                        'white_offset': white_offset
                    })

        except KeyboardInterrupt:
            print(f"子进程 {os.getpid()} 捕获到 KeyboardInterrupt，正在退出...")
            stop_event_2.set()
            if not queue_out.full():
                queue_out.put_nowait(None)
            break



@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop, frame_queue1, frame_queue1_1, frame_queue2, queue3_in, queue3_out, data_queue3


    main_loop = asyncio.get_running_loop()  # 获取主事件循环
    frame_queue1 = asyncio.Queue(maxsize=2)  # 只保留最新帧
    frame_queue1_1 = asyncio.Queue(maxsize=2)  # 只保留最新帧
    frame_queue2 = asyncio.Queue(maxsize=2)
    data_queue3 = asyncio.Queue(maxsize=2)
    asyncio.set_event_loop(main_loop)


    task1 = asyncio.create_task(camera_processing())
    

    yield
    print('开始关闭进程')
    stop_event.set()
    stop_event_2.set()


    if p.is_alive():
        p.terminate()
        p.join()
    print("子进程已终止")

    queue3_in.cancel_join_thread()
    queue3_out.cancel_join_thread()

# 修改摄像头处理线程函数


async def camera_processing():
    camera = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = camera.read()
        if ret:
            if not frame_queue1.full():
                frame_queue1.put_nowait(frame)
            if not frame_queue1_1.full():
                frame_queue1_1.put_nowait(frame)
            if not frame_queue2.full():
                frame_queue2.put_nowait(frame)
            if not queue3_in.full():
                queue3_in.put_nowait(frame)

        await asyncio.sleep(1/30)
            
    camera.release()






app = FastAPI(lifespan=lifespan)

app.mount("/web", StaticFiles(directory="web"), name="web")



@app.get("/")
async def read_root():
    return FileResponse("web/index.html")



@app.websocket("/ws1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while not stop_event.is_set():
            # 异步等待新帧（每秒最多30帧）
            frame = await frame_queue1.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            # await asyncio.sleep(1/30)  # 控制最大帧率
    except WebSocketDisconnect:
        print("客户端断开连接1")

@app.websocket("/ws1_1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while not stop_event.is_set():
            # 获取原始帧
            frame = await frame_queue1_1.get()
            
            # 应用zhifangtu预处理
            processed_frame = zhifangtu(frame)
            
            # 编码并发送图像
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send_bytes(buffer.tobytes())
            
    except WebSocketDisconnect:
        print("客户端断开连接1-1")


@app.websocket("/ws2")
async def websocket_endpoint(websocket: WebSocket):
    # global h_lower, h_upper, s_lower, s_upper, v_lower, v_upper
    await websocket.accept()

    try:
        while not stop_event.is_set():
            # 定义 HSV 阈值范围
            lower_bound = np.array([h_lower, s_lower, v_lower])
            upper_bound = np.array([h_upper, s_upper, v_upper])

            frame = await frame_queue2.get()

            frame = zhifangtu(frame)
            # 将 BGR 图像转换为 HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            ret, buffer = cv2.imencode('.jpg', mask)  # 将帧转换为 JPEG 格式
            frame = buffer.tobytes()
            await websocket.send_bytes(frame)
            # await asyncio.sleep(1/30)

    except WebSocketDisconnect as e:
        print("客户端断开连接2")

    except RuntimeError as e:
        print(e)


@app.websocket("/ws3")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    fps_counter = 0
    fps = 0
    last_time = time.time()
    
    try:
        while not stop_event.is_set():
            try:
                out_result = await main_loop.run_in_executor(None, queue3_out.get, True, 1)
                if out_result == None:
                    break
            except queue.Empty:
                continue

            frame = out_result['frame']

            if not data_queue3.full():
                data_result = {k: v for k, v in out_result.items() if k != 'frame'}
                data_queue3.put_nowait(data_result) 
            
            # 计算FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:  # 每秒更新一次
                fps = fps_counter / (current_time - last_time)
                fps_counter = 0
                last_time = current_time
                # 在图像上绘制FPS
                
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            await websocket.send_bytes(frame_bytes)
            
    except WebSocketDisconnect:
        print("客户端断开连接3")


# 创建一个任务来处理WebSocket消息



@app.websocket("/ws4")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def receive_hsv_values(websocket):
        global h_lower, h_upper, s_lower, s_upper, v_lower, v_upper
        while not stop_event.is_set():
            data = await websocket.receive_json()
            if "h_lower" in data:
                h_lower = min(max(data["h_lower"], 0), 180)
                h_upper = min(max(data["h_upper"], 0), 180)
                s_lower = min(max(data["s_lower"], 0), 255)
                s_upper = min(max(data["s_upper"], 0), 255)
                v_lower = min(max(data["v_lower"], 0), 255)
                v_upper = min(max(data["v_upper"], 0), 255)
                print(f"HSV thresholds updated: {h_lower}, {h_upper}, {s_lower}, {s_upper}, {v_lower}, {v_upper}")


    async def send_data(websocket):
        while not stop_event.is_set():
            data_result = await data_queue3.get()
            data_result['timestamp'] = time.time()
            # 只发送检测数据
            await websocket.send_json(data_result)


    # 启动接收任务
    try:
        await asyncio.gather(receive_hsv_values(websocket), send_data(websocket))
                
    except WebSocketDisconnect:
        print("客户端断开连接4")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 读取上传的文件内容
        contents = await file.read()
        # 将字节转换为numpy数组
        nparr = np.frombuffer(contents, np.uint8)
        # 解码为图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("无法解码图像")
            
        # 放入用户队列
        queue3_user_in.put(img)
        # 等待结果
        try:
            result = await main_loop.run_in_executor(None, queue3_user_out.get, True, 60)
            
            print(result['frame'].shape)
            # 编码结果图像为JPEG
            _, encoded_img = cv2.imencode('.jpg', result['frame'])
            
            return StreamingResponse(
                io.BytesIO(encoded_img.tobytes()),
                media_type="image/jpeg",
                headers={
                    "car_area": str(result.get('car_area', '')),
                    "car_offset": str(result.get('car_offset', '')),
                    "white_area": str(result.get('white_area', '')),
                    "white_offset": str(result.get('white_offset', ''))
                }
            )
        except queue.Empty:
            raise HTTPException(status_code=408, detail="处理超时")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))





if __name__ == '__main__':
    stop_event_2 = multiprocessing.Event()
    queue3_in = multiprocessing.Queue(maxsize=2)
    queue3_out = multiprocessing.Queue(maxsize=2)
    queue3_user_in = multiprocessing.Queue()
    queue3_user_out = multiprocessing.Queue()
    queue3_in.put_nowait(garbage_frame)

    p = multiprocessing.Process(target=inference_worker, args=(queue3_in, queue3_out, queue3_user_in, queue3_user_out, './model/yolo11_13k.engine'))
    p.start()

    stop_event = asyncio.Event()
    uvicorn.run(app, host='0.0.0.0', port=8000)
