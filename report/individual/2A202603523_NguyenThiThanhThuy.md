# Báo cáo Day 7 — Nguyễn Thị Thanh Thúy

**MSSV:** 2A202603523 · **Lớp:** L3A

**Nguồn Day 6 được dùng:** PDF do người học cung cấp; đường dẫn máy cục bộ không được đưa vào báo cáo công khai.

## Dữ liệu và benchmark

- Tài liệu NASA *Science 2020–2024: A Vision for Scientific Excellence*, bản cập nhật 2023 do người học cung cấp. PDF có 71 trang vật lý; trang 71 không có text trích xuất, 70 trang còn lại có text. Cần kiểm tra trang nguồn và quyền của nội dung bên thứ ba trước khi tái phân phối PDF; báo cáo không đính kèm bản PDF.
- Pipeline chia nội dung thành 240 chunks (tối đa 1.000 ký tự, overlap 120 ký tự), giữ trang PDF gốc. Embedding FastEmbed local/ONNX CPU: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, Apache-2.0. Vector store Qdrant embedded local, cosine, exact search (HNSW tắt), dựng từ PDF này. Đây là index tái tạo cục bộ, không phải database Qdrant đã lưu của Day 6.
- Benchmark [`data/nasa_science_plan_questions.json`](../../data/nasa_science_plan_questions.json): 29 câu, gồm 20 câu có evidence dev, 5 câu có evidence test, 2 câu thiếu evidence mỗi split. Test được chạy sau khi chốt cấu hình từ dev. Mọi gold quote của câu answerable đã được kiểm tra khớp text trích từ đúng trang PDF.
- Tiêu chí retrieval: Hit@1 và Evidence Hit@3 tính trên câu có evidence; Context Evidence Rate là số câu có evidence trong context cuối / số câu có evidence. Báo `n/N`, tỷ lệ; context budget tính bằng ký tự. Câu thiếu evidence không đưa vào mẫu số retrieval.
- Agent dùng OpenAI Chat Completions mặc định (`gpt-4o-mini`) hoặc Ollama local qua endpoint tương thích OpenAI. Không cấu hình OpenAI key; máy có Ollama với `qwen3-vl:2b`. Đã sửa để TF-IDF không dựng dense index không cần thiết, nguyên nhân lần thử trước gây lỗi cấp phát ONNX. Tuy nhiên chưa có câu trả lời agent NASA được xác nhận: request Ollama riêng vẫn hết timeout 300 giây.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter/rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | Dense Qdrant; k=3; không filter/rerank; 1.600 ký tự | Baseline vector Day 7 tái tạo từ PDF | dev | 5/20 (25%) | 7/20 (35%) | 5/20 (25%) | median 17.25 ms/query; $0 |
| Phương pháp 2 | TF-IDF; k=3; 1.600 ký tự | Lexical baseline chạy local | dev | 15/20 (75%) | 17/20 (85%) | 15/20 (75%) | median 1.34 ms/query; $0 |
| Phương pháp 3 | Hybrid RRF dense + TF-IDF; k=3; 1.600 ký tự | Kết hợp hai ranking có RRF | dev | 8/20 (40%) | 14/20 (70%) | 9/20 (45%) | median 18.24 ms/query; $0 |
| Cải thiện 1 | TF-IDF; k=3; 2.400 ký tự | Tăng context budget để không bỏ evidence nằm ở chunk xếp sau | dev | 15/20 (75%) | 17/20 (85%) | 17/20 (85%) | median 1.34 ms/query; $0 |
| Cải thiện 2 / cấu hình chốt | TF-IDF; k=5; 4.000 ký tự | Tăng số đoạn ứng viên và ngân sách context | dev | 15/20 (75%) | 17/20 (85%) | 19/20 (95%) | median 1.33 ms/query; $0 |
| Baseline | Dense Qdrant; k=3; 4.000 ký tự | Chạy baseline sau khi chốt cấu hình | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | median 26.57 ms/query; $0 |
| Cấu hình đã chọn | TF-IDF; k=5; 4.000 ký tự | Chọn theo Context Evidence Rate dev; test chỉ chạy sau khi chốt | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | median 1.49 ms/query; $0 |

Chi phí embedding/retrieval API là $0 vì chạy local. Lượt xác nhận hiện tại tái sử dụng Qdrant index; thời gian khởi tạo/mở index là 6.61 giây trên dev và 6.71 giây trên test. Thời gian median có thể thay đổi theo máy và cache. Test chỉ có 5 câu answerable nên tỷ lệ 5/5 không phải bằng chứng tổng quát về chất lượng. Kết quả test retrieval chưa bao gồm chất lượng câu trả lời của agent.

**Phân tích lỗi dev và thay đổi đã thử**

1. Câu `d01` hỏi nguồn định hướng ưu tiên. TF-IDF xếp chunk evidence trang 4 ở vị trí 3–4; với k=5 nhưng budget 4.000 ký tự, context packer chỉ giữ ba chunk đầu nên bỏ mất đoạn đúng. Tăng budget từ 1.600 lên 2.400 cải thiện context evidence từ 15/20 lên 17/20; tăng k lên 5 và budget 4.000 đạt 19/20. Câu này vẫn là lỗi còn lại của cấu hình chốt.
2. Câu `d08` về kế hoạch Artemis tới năm 2024 bị dense retrieval xếp các trang không liên quan lên đầu (dev Hit@3 không có evidence); TF-IDF tìm thấy đoạn đúng ở trang 16. Điều này ủng hộ chọn lexical cho cấu hình cuối. Hybrid RRF cũng kém TF-IDF trên dev (14/20 so với 17/20 Evidence Hit@3), nên không chọn hybrid.

**Audit thủ công context test** (điểm 0/1/2 theo BENCHMARK.md):

| Query | Trang evidence | Dense baseline | TF-IDF đã chọn | Lý do |
|---|---:|---:|---:|---|
| `t01` | 30 | 2 | 2 | Context có đoạn về 614 sự kiện, kèm tên nguồn và trang. |
| `t02` | 37 | 2 | 2 | Context có số giao hàng CLPS và tổng payload, kèm nguồn/trang. |
| `t03` | 39 | 2 | 2 | Context có mốc KDP-B, kèm nguồn/trang. |
| `t04` | 40 | 2 | 2 | Context có số ấn phẩm, kèm nguồn/trang. |
| `t05` | 61 | 2 | 2 | Context có số người đăng ký workshop, kèm nguồn/trang. |

Cả hai cấu hình đều đạt 5/5 context đủ evidence; `t06` chỉ có thông tin CLPS liên quan nhưng không có giá từng task order, `t07` có nhắc ISS nhưng không có độ cao quỹ đạo. Hai câu này không đưa vào mẫu số retrieval/context answerable; cần kiểm tra abstention của agent khi có API.

## Agent và đánh giá

- Tool search do model gọi; cấu hình dự kiến tối đa 3 lượt search, method TF-IDF, top-k=5, context budget 4.000 ký tự. Câu trả lời cần citation dạng `[chunk-id]`; loop kiểm tra ID citation có trong context và từ chối khi thiếu citation/không có context. Kiểm tra ID không thay thế audit xem nguồn có hỗ trợ câu trả lời.
- **Chưa có kết quả agent NASA**: không cấu hình OpenAI key. Lỗi dựng dense index cho TF-IDF đã được sửa. Lần thử agent một câu sau sửa vẫn hết timeout Ollama sau 300 giây; chưa có trace để chấm. Không suy ra các metric agent từ retrieval.
- Retrieval traces và summary: `starter/results/nasa-science-plan/dev/` và `starter/results/nasa-science-plan/test/` (local, bị Git ignore). Trace test cho câu thiếu evidence cho thấy `t06` tìm được thông tin gần chủ đề CLPS nhưng không có giá từng task order; `t07` không tìm thấy evidence về độ cao quỹ đạo ISS. Cần xác nhận hành vi abstention qua agent và audit thủ công.
- Chạy retrieval không cần API. Cài dependency bằng `python -m pip install -r starter/requirements-day7.txt`; hướng dẫn lệnh và thiết lập key nằm trong [`starter/README.md`](../../starter/README.md). Chỉ nạp key khi dùng `--mode agent`; không commit file `.env` hoặc gửi key qua chat.

## Thử thách nâng cao

### Tài liệu, xử lý và benchmark

- **Nguồn:** [CASSINI. Report on the Phase A study: Saturn Orbiter and Titan probe](https://ntrs.nasa.gov/citations/19910008874), NASA NTRS report 19910008874, tháng 10/1988. Metadata NTRS ghi `PUBLIC`, `GOV_PUBLIC_USE_PERMITTED` và không có cờ third-party material. Giữ attribution; kiểm tra notice từng trang trước khi tái phân phối PDF.
- **Bản local:** Được lưu ngoài repository và không copy vào Git. PDF có 148 trang, 146 trang có text trích xuất, tổng 355.668 ký tự; p4 và p148 là scan trắng, không có text. Cả 148 trang có embedded scan; đã xuất 148 JPEG để vision model có thể kiểm tra vùng gốc.
- **Index:** 463 chunks, tối đa 1.000 ký tự, overlap 120; FastEmbed `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều; Qdrant embedded local. Pipeline trích toàn bộ text có thể đọc và giữ số trang; không chạy OCR/layout mới, không tạo multimodal embedding. Khi câu hỏi nêu mã Figure/Table, pipeline đưa chunk caption tương ứng lên trước và gửi tối đa ba ảnh trang đã retrieve cho model vision-capable. Vì vậy ảnh scan chỉ đến được model nếu caption/text retrieval tìm được trang đó.
- **Benchmark:** [`data/cassini_advanced_questions.json`](../../data/cassini_advanced_questions.json): dev 18 (15 answerable, 3 missing); test 30 (24 answerable, 6 missing): table 6, image/diagram 5, chart 5, multi-hop 4, text 4. Mỗi gold quote đã được đối chiếu với text layer của đúng trang PDF; vùng evidence trong bảng/ảnh/biểu đồ đã được xác định từ scan gốc. Multi-hop `evidence_mode=all` yêu cầu đủ mọi gold item.
- **Tái tạo:** page images, chunks, Qdrant index, retrieval summary/traces được lưu cục bộ trong `starter/results/cassini-advanced/`. Nguồn và cấu hình chi tiết nằm trong [`data/advanced_cassini_manifest.json`](../../data/advanced_cassini_manifest.json).

### Thử nghiệm retrieval

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence | Median retrieval |
|---|---|---:|---:|---:|---:|
| Dense, k=3, budget=1600 | dev | 10/15 (66.7%) | 14/15 (93.3%) | 12/15 (80.0%) | 47.99 ms/query |
| TF-IDF, k=3, budget=1600 | dev | 9/15 (60.0%) | 14/15 (93.3%) | 12/15 (80.0%) | 21.95 ms/query |
| TF-IDF, k=3, budget=2400 | dev | 9/15 (60.0%) | 14/15 (93.3%) | 14/15 (93.3%) | 22.42 ms/query |
| TF-IDF, k=5, budget=4000 — **đã chọn** | dev | 9/15 (60.0%) | 14/15 (93.3%) | 15/15 (100%) | 24.00 ms/query |
| TF-IDF, k=8, budget=6000 | dev | 9/15 (60.0%) | 14/15 (93.3%) | 15/15 (100%) | 21.99 ms/query |
| Hybrid, k=5, budget=4000 | dev | 10/15 (66.7%) | 14/15 (93.3%) | 14/15 (93.3%) | 44.59 ms/query |
| TF-IDF, k=5, budget=4000 — **chọn từ dev, khóa cho test** | test | 20/24 (83.3%) | 24/24 (100%) | 24/24 (100%) | 22.48 ms/query |
| Dense baseline, k=3, budget=4000 | test | 17/24 (70.8%) | 21/24 (87.5%) | 21/24 (87.5%) | 48.31 ms/query |

Retrieval và embedding chạy cục bộ nên API cost là $0. Lần build Qdrant đầu tiên mất 52.84 giây; các lần mở lại index đã có mất 6.64–7.66 giây. Việc bổ sung định tuyến caption Figure/Table cải thiện khả năng đưa trang scan liên quan vào context. TF-IDF k=5/budget=4000 đạt 15/15 trên dev, bằng TF-IDF k=8/budget=6000 nhưng dùng context ít hơn nên được chọn. Test retrieval được chạy một lần sau khi khóa lựa chọn bằng `--finalize`. Kết quả 24/24 trên test chỉ là **retrieval evidence** của các câu có đáp án; sáu câu thiếu evidence không nằm trong mẫu số và đây không phải chất lượng câu trả lời của agent.

### Đánh giá agent và việc còn lại

- **Lần thử agent dev ban đầu** bằng Ollama local `qwen3-vl:2b` (không dùng API key) lỗi 18/18 câu ở timeout mặc định 60 giây. Đã sửa các nguyên nhân phía pipeline: không dựng dense index cho TF-IDF, ép search ở lượt đầu, giới hạn ảnh local còn một trang và thu nhỏ tối đa 1.024 px; mặc định 2.048 token đầu ra và timeout 300 giây. Chẩn đoán xác nhận ảnh gốc làm prompt vượt context mặc định 4.096 token. Một bước tool-call trước đó đã chạy được nhưng mất 216,5 giây và chưa tạo câu trả lời; lần chạy agent NASA riêng vẫn timeout 300 giây. Kiểm tra mới với câu Cassini `cd01` trả được truy vấn search sau 234,5 giây; context có chunk trang 42 chứa evidence, nhưng bước sinh câu trả lời không kết thúc sau hơn 300 giây và phải dừng thủ công. Do đó chưa có câu trả lời agent để chấm.
- **Không chạy agent test** vì chưa có câu trả lời dev để audit/chốt; tiếp tục test trong trạng thái này sẽ không đáp ứng quy trình dev→test của ADVANCED.md. Chưa có nhãn grounded answer, citation support, abstention hoặc tỷ lệ đạt; không thể kết luận đạt bonus 90%. Cần chạy lại dev với phần cứng/model phù hợp hơn rồi audit trước khi chạy test.
- Trong `reviews-test-*.json`, người chấm cần điền nhãn cho toàn bộ 30 trường hợp: `answer_grade` và `citation_support` cho câu có đáp án; `abstention_correct` cùng lý do cho câu thiếu evidence. `benchmark_pass` và `by_group` trong `agent_metrics.py` tính cả 30 câu và lỗi hệ thống trong mẫu số. Chỉ báo cáo đạt 27/30 (90%) nếu audit thủ công xác nhận.
- **Giới hạn:** Báo cáo Phase A năm 1988 là tài liệu thiết kế, không phải báo cáo kết quả bay Cassini. Đã lưu scan của toàn bộ 148 trang nhưng chưa có image/OCR retrieval độc lập; hai trang scan trắng không có text evidence. pypdf phát cảnh báo metadata `/Length` trùng lặp, nhưng trích xuất được 148 ảnh và tạo 463 chunk thành công.
