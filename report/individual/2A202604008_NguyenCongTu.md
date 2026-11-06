# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604008
- **Họ và tên:** Nguyễn Công Tú
- **SEED** (in ở ô đầu notebook): 7033

## 1. Bằng chứng

| Tệp | Có? |
|---|:---:|
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5:

| Phương pháp | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg | 0.918 | 0.947 | 0.161 | 0.000 |
| 2. BoW + MLP | 0.947 | 0.974 | 0.463 | 0.000 |
| 3a. MiniLM + kNN | 0.889 | 0.927 | 121.689 | 0.000 |
| 3b. MiniLM + LogReg | 0.918 | 0.974 | 34.245 | 0.000 |
| 3c. MiniLM + MLP head | 0.955 | 0.974 | 36.121 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.927 | 0.927 | 2762.569 | 0.000 |
| 4b. LLM lớn zero-shot | 0.927 | 0.927 | 2346.800 | 0.000 |
| 4c. LLM nhỏ few-shot | 0.952 | 0.952 | 2600.626 | 0.000 |

## 3. Dự đoán và thực tế

### Dự đoán 1 – Encoder sẽ tốt hơn BoW ở tin nhắn ngụy trang (emb_tricky)

- **Dự đoán:** MiniLM + LogReg sẽ phân loại đúng nhiều hơn BoW.
- **Thực tế:** BoW đúng 5/6, Encoder + LogReg đúng 4/6.
- **Lý do:** Embedding có khả năng biểu diễn ngữ nghĩa nhưng không được huấn luyện chuyên biệt để nhận diện spam. Một số cách diễn đạt spam lạ như “Your mobile number has been chosen...” chưa được mô hình nhận diện tốt.

### Dự đoán 2 – Embedding + LogReg và LLM Few-shot sẽ nằm trên đường Pareto (tradeoff)

- **Dự đoán:** BoW + LogReg, Embedding + LogReg và LLM Few-shot nằm trên đường Pareto.
- **Thực tế:** Chỉ có BoW + LogReg và BoW + MLP nằm trên đường Pareto.
- **Lý do:** BoW + MLP có F1 SUB cao hơn nhưng vẫn rất nhanh. MiniLM và LLM có độ trễ cao hơn, trong khi không tạo ra lợi thế đủ lớn về chất lượng để nằm trên đường Pareto.

### Dự đoán 3 – LLM sẽ khó xử lý prompt injection nhất (hard_eval)

- **Dự đoán:** LLM dễ bị đánh lừa bởi các câu lệnh cố tình thao túng trong tin nhắn; accuracy stress test khoảng 50–70%.
- **Thực tế:** LLM zero-shot đạt accuracy 88%, còn LLM dùng prompt phòng thủ đạt 92%. Riêng nhóm prompt injection, mô hình phòng thủ đạt 100%.
- **Lý do:** LLM có khả năng hiểu ngữ cảnh và phân biệt nội dung tin nhắn với chỉ dẫn. Prompt phòng thủ giúp nhấn mạnh rằng nội dung SMS là dữ liệu không đáng tin cậy. Tuy nhiên, prompt phòng thủ vẫn có thể làm giảm hiệu quả ở một số nhóm khác.

## 4. Phân tích 50 case khó

### Model tụt nhiều nhất

Model tụt nhiều nhất: **BoW + Logistic Regression**.

- F1 trên SUB 40: 0.947.
- F1 trên 50 case khó: 0.378.
- Mức giảm: 0.569 điểm F1, lớn nhất trong các mô hình.
- BoW + MLP cũng giảm mạnh, từ 0.974 xuống 0.421.

### Nhóm case gây khó khăn

- BoW + LogReg chỉ đạt 12% ở nhóm Manipulation (tin nhắn thao túng).
- Đạt 17% ở nhóm Obfuscation (né từ khóa).
- Đạt 20% ở nhóm Natural spam (spam giọng tự nhiên).
- BoW + MLP đặc biệt yếu ở nhóm Obfuscation, chỉ đạt 0%.

### Hai ví dụ tiêu biểu

**Case #21:**

> Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x

- BoW chủ yếu nhìn các từ riêng lẻ, không hiểu được ý đồ mạo danh và thao túng tâm lý.
- Tin nhắn không chứa nhiều từ khóa spam quen thuộc nên dễ bị bỏ sót.

**Case #25:**

> This is your manager, I'm in a meeting and can't talk. Buy 5 Amazon gift cards and send me the codes ASAP, I'll reimburse you

- Mô hình BoW khó nhận diện mối liên hệ giữa việc mua gift card, gửi mã và hành vi lừa đảo nếu các từ này không có trọng số spam đủ cao.

## 5. Thử thách cá nhân

### Đề bài

Giảm vocabulary còn 100 từ và tăng lên 5000 từ bằng `build_vocab(..., max_size)`. Train lại và báo F1 test.

### Cách thực hiện

- Tạo vocabulary mới với `max_size=100`, vector hóa dữ liệu train/test và huấn luyện lại Logistic Regression.
- Lặp lại quy trình với `max_size=5000`.
- Giữ nguyên tập train/test và cách đánh giá để so sánh công bằng.

### Kết quả

| Vocabulary | F1 test |
|---|---:|
| 100 từ | 0.8024 |
| 5000 từ | 0.9180 |

### Giải thích

- Với 100 từ, mô hình bị giới hạn đặc trưng, nhiều từ khóa quan trọng liên quan đến spam bị loại bỏ nên khả năng phân loại giảm.
- Với 5000 từ, mô hình có thêm thông tin về nội dung tin nhắn, nhận diện được nhiều dấu hiệu spam hơn nên F1 tăng 0.1156, tương đương 11.56 điểm phần trăm.

## 6. Phương án cho sản phẩm thật

### Lựa chọn mô hình

Em chọn **BoW + MLP** nếu ưu tiên cân bằng giữa tốc độ và chất lượng phân loại, vì mô hình đáp ứng yêu cầu SLA p95 dưới 50 ms/tin và có F1 test cao hơn BoW + LogReg.

### Các số liệu so sánh

| Tiêu chí | BoW + LogReg | BoW + MLP |
|---|---:|---:|
| F1 test | 0.918 | 0.947 |
| p95 latency | 0.161 ms | 0.463 ms |
| Throughput | 21.292 tin/s | 28.921 tin/s |

### Đánh đổi chấp nhận

- **Chọn BoW + MLP:** Chấp nhận mô hình có nhiều tham số hơn Logistic Regression và có nguy cơ overfitting cao hơn.
- **Hạn chế về ngữ cảnh:** BoW khó hiểu ngữ cảnh, dễ nhầm các tin nhắn có từ khóa giống spam nhưng thực tế là ham.
- **Khả năng chống spam biến thể:** Qua stress test, BoW + MLP chỉ đạt F1 0.421 trên 50 tin nhắn khó, cho thấy khả năng chống spam biến thể và cách viết đánh lừa còn hạn chế.

## 7. AI log và điều đã học

### Nhật ký sử dụng AI

| Lần dùng AI | Em đã hỏi gì? | AI hỗ trợ gì? | Em đã sửa/kiểm tra gì? |
|---|---|---|---|
| 1 | Giải thích công thức cập nhật trọng số Logistic Regression và lỗi của BoW | Giải thích gradient, learning rate và ví dụ phân loại nhầm | Đối chiếu với kết quả và ví dụ SMS trong notebook |
| 2 | Hoàn thiện prompt, parse nhãn và gọi LLM | Gợi ý cách tạo prompt zero-shot/few-shot và tính latency, token | Sửa cách lưu response, đếm output token và kiểm tra nhãn dự đoán |
| 3 | Phân tích Pareto, SLA và stress test | Hỗ trợ so sánh F1, latency và tìm nhóm dữ liệu mô hình yếu | Thay dự đoán ban đầu bằng số liệu thực nghiệm; phân biệt F1 với accuracy |
| 4 | Chọn mô hình cho sản phẩm thật | Hỗ trợ cân nhắc chất lượng, tốc độ và rủi ro | Chọn phương án dựa trên SLA và kết quả stress test |
| 5 | Nhờ AI hỗ trợ chỉnh sửa file Markdown (.md) cho bài tập | AI hỗ trợ tổng hợp câu trả lời từ Phần 1 đến Phần 7, chỉnh sửa nội dung, định dạng tiêu đề, bảng và code để dễ đọc | Em kiểm tra lại nội dung, đối chiếu số liệu với notebook và chỉnh sửa những chỗ chưa chính xác trước khi hoàn thiện file |

### Điều em hiểu ra hôm nay:

Trước đây em nghĩ mô hình có F1 cao nhất sẽ luôn là mô hình tốt nhất, nhưng thực tế còn phải xét tốc độ xử lý, chi phí và khả năng hoạt động với dữ liệu khó. Em hiểu rằng BoW có tốc độ nhanh nhưng hạn chế trong việc hiểu ngữ cảnh, còn Embedding và LLM có khả năng nắm bắt ngữ nghĩa tốt hơn nhưng cần nhiều tài nguyên và có độ trễ cao hơn. Vì vậy, khi chọn mô hình cho sản phẩm thật, cần cân bằng giữa độ chính xác, tốc độ và khả năng hoạt động ổn định trong thực tế.