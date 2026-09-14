Đề tài "TEETH SEGMENTATION IN PANORAMIC DENTAL X-RAY USING MASK REGIONAL CONVOLUTION NEURAL NETWORK " tập trung vào việc ứng dụng các kỹ thuật Deep Learning để giải quyết bài toán thị giác máy tính trong lĩnh vực y tế. 

Việc lựa chọn đề tài này xuất phát từ ba lý do chính:

**1.	Tính thực tiễn và cấp thiết trong y tế**: Trong nha khoa hiện đại, dữ liệu hình ảnh đang ngày càng lớn. Việc phân đoạn răng thủ công không chỉ tốn thời gian mà còn phụ thuộc vào chủ quan của bác sĩ, dễ dẫn đến sai sót. Một hệ thống tự động phân đoạn chính xác sẽ giúp:

o	Hỗ trợ bác sĩ lập kế hoạch điều trị nhanh chóng hơn.

o	Cung cấp dữ liệu đầu vào cho các hệ thống định danh răng tự động. 

o	Phát hiện sớm các bất thường về cấu trúc hoặc số lượng răng.

<img width="846" height="488" alt="image" src="https://github.com/user-attachments/assets/5ba4bc9c-d243-4a87-ac62-8322b063ecd6" />

                                `Kết quả đầu ra của mô hình đối với một ảnh trong tập dự đoán`

**2.	Độ khó và tính khoa học của bài toán**: Phân đoạn răng trên ảnh X-quang khó hơn so với các đối tượng thông thường do:

o	Độ tương phản thấp giữa răng và xương hàm.

o	Sự chồng lấn giữa các chân răng hoặc tình trạng răng mọc chen chúc.

o	Sự xuất hiện của các vật liệu nhân tạo (răng giả, vít chỉnh nha) làm nhiễu mô hình. Việc giải quyết được những khó khăn này mang lại giá trị khoa học cao trong lĩnh vực xử lý ảnh y tế.

• Dữ liệu gốc: https://tdd.ece.tufts.edu/Tufts_Dental_Database/

• Dữ liệu đã xử lý: https://vnshort.com/I60j
<img width="940" height="386" alt="image" src="https://github.com/user-attachments/assets/53b19fad-60b3-4a4f-b42a-162785742239" />

**3.	Khả năng mở rộng**: Công nghệ phân đoạn thực thể (Instance Segmentation) không chỉ dừng lại ở răng mà có thể mở rộng sang phân đoạn xương hàm, khối u hoặc các bệnh lý vùng mặt, mở ra hướng phát triển bền vững cho nghiên cứu.

Thông qua quá trình thực hiện đồ án, em đặt ra các mục tiêu cụ thể sau:

•	Làm chủ lý thuyết: Hiểu sâu về các kiến trúc mạng nơ-ron tích chập (CNN) và sự tiến hóa của các dòng mạng phát hiện đối tượng từ R-CNN, Fast R-CNN, Faster R-CNN đến Mask R-CNN.

•	Kỹ năng thực hành: Thành thạo các kỹ năng tiền xử lý dữ liệu y tế (xử lý nhiễu, cân bằng độ tương phản), kỹ thuật dán nhãn (labeling) và sử dụng các thư viện học sâu phổ biến (như PyTorch hoặc TensorFlow/Keras).

•	Tư duy giải quyết vấn đề: Học cách tinh chỉnh siêu tham số (hyperparameter tuning), áp dụng Transfer Learning và đánh giá mô hình bằng các chỉ số chuyên biệt (Dice Coefficient, mAP, Accuracy).

•	Tiếp cận quy trình nghiên cứu: Trải nghiệm quy trình hoàn chỉnh từ việc tìm hiểu bài toán, chuẩn bị dữ liệu, xây dựng mô hình đến thực nghiệm và viết báo cáo khoa học.

`Sơ đồ mô hình`
<img width="850" height="461" alt="image" src="https://github.com/user-attachments/assets/9b0730a2-9a8b-4a4b-b787-2f9d2875dca9" />

Ảnh X-quang toàn cảnh (Panoramic X-ray) là một công cụ chẩn đoán phổ biến trong nha khoa, cho phép bác sĩ quan sát toàn bộ cấu trúc hàm, răng và các mô liên quan. Tuy nhiên, việc phân tích thủ công hàng nghìn pixel để tách biệt chính xác từng đơn vị răng là một thách thức lớn. Đề tài này đề xuất sử dụng kiến trúc Mask R-CNN nhằm tự động hóa quá trình nhận diện và phân đoạn mặt nạ (mask) cho từng chiếc răng, tạo cơ sở cho các hệ thống hỗ trợ chẩn đoán thông minh.
