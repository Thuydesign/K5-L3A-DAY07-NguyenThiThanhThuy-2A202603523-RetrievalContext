# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day7_pipeline.py` | Trích PDF, benchmark dense / TF-IDF / hybrid và chạy agent OpenAI tùy chọn |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## Chạy trên PDF NASA Science Plan

Benchmark mẫu [`data/nasa_science_plan_questions.json`](../data/nasa_science_plan_questions.json)
có 25 câu có evidence (20 dev / 5 test) và 4 câu thiếu evidence (2 mỗi split). Đây là benchmark khởi đầu:
hãy kiểm tra độc lập câu hỏi, đáp án và evidence trước khi nộp. PDF do người học cung cấp
không nằm trong repo; script giữ nguyên số trang PDF (trang đầu là 1).

Cài các dependency chỉ dành cho pipeline PDF/retrieval:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r starter\requirements-day7.txt
```

Chạy retrieval trên dev trước; lệnh này trích PDF, dựng Qdrant local bằng model embedding
FastEmbed tương thích 384 chiều, rồi so sánh dense, TF-IDF, hybrid cùng các cấu hình top-k/context budget khác nhau. Không cần API key:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\2020-2024-nasa-science-plan-yr-23-update-final.pdf" --mode retrieval --split dev
```

Kết quả và chunks được ghi dưới `starter/results/nasa-science-plan/` (đã ignore bởi Git).
Đọc các lỗi dev, chốt `--selected-method`, `--selected-top-k` và
`--selected-context-budget`, rồi mới chạy test. `--finalize` là xác nhận đã chốt cấu hình;
test chỉ chạy baseline dense và cấu hình được chọn:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\2020-2024-nasa-science-plan-yr-23-update-final.pdf" --mode retrieval --split test --finalize --selected-method tfidf --selected-top-k 5 --selected-context-budget 4000
```

Mặc định agent dùng OpenAI Chat Completions và cần API key. Cũng có thể chạy agent hoàn toàn
cục bộ bằng Ollama qua OpenAI-compatible endpoint, không cần OpenAI key. Sao chép
`.env.example` thành `starter/.env` (bị Git ignore); không commit hay gửi key qua chat.

Với Ollama, cài [Ollama](https://ollama.com/download) 0.12.7 trở lên và tải model
vision/tool-calling [qwen3-vl:2b](https://ollama.com/library/qwen3-vl:2b) một lần:

```powershell
ollama pull qwen3-vl:2b
```

Với máy cấu hình thấp, bản 2B cần khoảng 1.9 GB dung lượng tải về; inference chạy trên máy của bạn.
Có thể đổi sang bản lớn hơn nếu máy đủ RAM/VRAM. Trong
`starter/.env`, để trống `OPENAI_API_KEY` và đặt:

```dotenv
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3-vl:2b
```

Khi `OPENAI_BASE_URL` trỏ loopback (`localhost`/127.0.0.1/::1), script bỏ qua API key và
không gửi header Authorization. Ollama cần đang chạy; các lệnh `--mode agent` bên dưới
giữ nguyên. Tài liệu Day 7 chỉ gửi câu hỏi, trích đoạn và ảnh tới endpoint đã cấu hình.
Sau khi chọn cấu hình retrieval trên dev, chạy agent với cấu hình đã chọn trên dev và đọc
kết quả trước. Mỗi run có tên riêng để không ghi đè trace; đồng thời tạo file
`reviews-<split>-<run-name>.json` với nhãn trống cần người học điền thủ công:

```powershell
if (-not (Test-Path starter\.env)) { Copy-Item starter\.env.example starter\.env }
# OpenAI cần key; Ollama local dùng OPENAI_BASE_URL/OPENAI_MODEL ở trên, không cần key.
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\2020-2024-nasa-science-plan-yr-23-update-final.pdf" --mode agent --split dev --method tfidf --top-k 5 --context-budget 4000 --run-name selected-tfidf
```

Chỉ sau khi đã đọc/chấm kết quả agent dev và chốt cấu hình, chạy baseline dense và
cấu hình đã chọn trên test. `--finalize` là xác nhận tường minh; chương trình từ chối
chạy agent trên test nếu thiếu cờ này. Các lần chạy tạo file riêng:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\2020-2024-nasa-science-plan-yr-23-update-final.pdf" --mode agent --split test --method dense --top-k 3 --context-budget 4000 --finalize --run-name baseline-dense
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\2020-2024-nasa-science-plan-yr-23-update-final.pdf" --mode agent --split test --method tfidf --top-k 5 --context-budget 4000 --finalize --run-name selected-tfidf
```

Sau mỗi run, đối chiếu thủ công câu trả lời, evidence, citation, từ chối và lỗi với PDF;
điền các trường `null` trong file `reviews-*.json`. Sau đó tổng hợp trực tiếp từ benchmark,
trace và reviews đã điền:

```powershell
.\.venv\Scripts\python.exe starter\agent_metrics.py --cases data\nasa_science_plan_questions.json --traces starter\results\nasa-science-plan\agent-test-selected-tfidf.json --reviews starter\results\nasa-science-plan\reviews-test-selected-tfidf.json --split test --output starter\results\nasa-science-plan\summary-test-selected-tfidf.json
.\.venv\Scripts\python.exe starter\agent_metrics.py --cases data\nasa_science_plan_questions.json --traces starter\results\nasa-science-plan\agent-test-baseline-dense.json --reviews starter\results\nasa-science-plan\reviews-test-baseline-dense.json --split test --output starter\results\nasa-science-plan\summary-test-baseline-dense.json
```

Trace ghi token usage tổng và theo từng câu; chi phí USD chưa tự tính vì giá model thay
đổi, cần đối chiếu usage với bảng giá tại ngày chạy. Script không gửi API request trong
retrieval mode. Citation ID hợp lệ không tự chứng minh nội dung được nguồn hỗ trợ.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.

## Thử thách nâng cao Cassini

Benchmark [`data/cassini_advanced_questions.json`](../data/cassini_advanced_questions.json)
gồm 18 câu dev (15 có đáp án, 3 thiếu evidence) và 30 câu test (24 có đáp án, 6 thiếu
evidence). Test có đúng các nhóm tối thiểu: 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop
và 4 câu text. Multi-hop yêu cầu evidence từ tất cả các vùng được liệt kê.

Nguồn là báo cáo Phase A của NASA, NTRS 19910008874, tháng 10/1988. Manifest
[`data/advanced_cassini_manifest.json`](../data/advanced_cassini_manifest.json) ghi nguồn,
quyền sử dụng, giới hạn và cấu hình. PDF không nằm trong repo; dùng bản đã tải trong Downloads
như đường dẫn ở lệnh bên dưới, hoặc thay bằng đường dẫn PDF của bạn. Pipeline index 146 trang
thành 463 chunk và xuất scan của cả 148 trang thành JPEG. Trang 4 và 148 là scan trắng. Nếu câu
hỏi nêu mã Figure/Table, pipeline ưu tiên chunk caption tương ứng và đính kèm ảnh của tối đa ba
trang tìm được vào yêu cầu gửi vision model. Đây không phải image embedding hay OCR độc lập;
vision model chỉ nhận trang scan sau khi retrieval theo text/caption tìm được trang đó.

Retrieval và lập chỉ mục cục bộ không cần API key. `--image-dir` lưu page scan:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\19910008874-Cassini-Saturn-Orbiter-Titan-Probe-Phase-A-Report.pdf" --questions data\cassini_advanced_questions.json --output starter\results\cassini-advanced --source-id cassini_phase_a_19910008874 --image-dir starter\results\cassini-advanced\page-images --mode retrieval --split dev
```

Trên dev, cấu hình được chọn là TF-IDF, top-k=5, context 4.000 ký tự: evidence của 15/15 câu
có đáp án đã vào context cuối. TF-IDF top-k=8 / 6.000 ký tự cũng đạt 15/15 nhưng đưa thêm
context không cần thiết, nên chọn top-k=5. Sau khi xem kết quả dev và khóa cấu hình, chạy test
retrieval đúng một lần:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\19910008874-Cassini-Saturn-Orbiter-Titan-Probe-Phase-A-Report.pdf" --questions data\cassini_advanced_questions.json --output starter\results\cassini-advanced --source-id cassini_phase_a_19910008874 --image-dir starter\results\cassini-advanced\page-images --mode retrieval --split test --finalize --selected-method tfidf --selected-top-k 5 --selected-context-budget 4000
```

Test retrieval đã khóa đưa evidence của 24/24 câu có đáp án vào context. Đây chỉ là metric
retrieval, không chứng minh chất lượng câu trả lời, citation, abstention hay điều kiện bonus 90%.

Trước khi chạy agent, cấu hình OpenAI key hoặc Ollama local trong `starter/.env` theo mục 1
(không gửi key qua chat, không ghi vào Git). Với Cassini, chọn model vision như `qwen3-vl:2b`
để model đọc được ảnh scan. Chạy dev trước, đối chiếu từng câu trả lời và citation với trang gốc;
chỉ điều chỉnh prompt/cấu hình dựa trên dev:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\19910008874-Cassini-Saturn-Orbiter-Titan-Probe-Phase-A-Report.pdf" --questions data\cassini_advanced_questions.json --output starter\results\cassini-advanced --source-id cassini_phase_a_19910008874 --image-dir starter\results\cassini-advanced\page-images --mode agent --split dev --method tfidf --top-k 5 --context-budget 4000 --run-name cassini-tfidf-dev
```

Mỗi request mặc định có timeout 300 giây và tối đa 2.048 token đầu ra. Agent ép gọi tool
search ở lượt chưa có evidence. Với Ollama, ảnh trang được thu nhỏ tối đa 1.024 px và chỉ
gửi một trang mỗi request để vừa context mặc định; endpoint từ xa giữ ảnh gốc và nhận tối
đa ba trang.
Có thể chỉnh timeout bằng `--request-timeout` và giới hạn đầu ra bằng
`--max-output-tokens`. Cả hai giá trị được ghi trong trace report.

Chỉ sau khi audit dev mới chạy agent test với `--finalize`. Trong file `reviews-test-*.json`,
người chấm phải điền `answer_grade` cho mỗi câu có đáp án (câu đúng và đầy đủ được 2), nhãn
`citation_support` cho từng citation; với câu thiếu evidence, điền `abstention_correct` và lý do:

```powershell
.\.venv\Scripts\python.exe starter\day7_pipeline.py --pdf "$env:USERPROFILE\Downloads\19910008874-Cassini-Saturn-Orbiter-Titan-Probe-Phase-A-Report.pdf" --questions data\cassini_advanced_questions.json --output starter\results\cassini-advanced --source-id cassini_phase_a_19910008874 --image-dir starter\results\cassini-advanced\page-images --mode agent --split test --method tfidf --top-k 5 --context-budget 4000 --finalize --run-name cassini-tfidf-final
.\.venv\Scripts\python.exe starter\agent_metrics.py --cases data\cassini_advanced_questions.json --traces starter\results\cassini-advanced\agent-test-cassini-tfidf-final.json --reviews starter\results\cassini-advanced\reviews-test-cassini-tfidf-final.json --split test --output starter\results\cassini-advanced\summary-test-cassini-tfidf-final.json
```

Metric cuối `benchmark_pass` là tỷ lệ đạt/không đạt theo rubric; lỗi provider và trường hợp
không có trace vẫn nằm trong mẫu số. `by_group` báo n/N theo nhóm. Chưa chạy agent test thật và
chưa audit thủ công 30 câu nên chưa thể công bố đạt 90%. Scan, chunks, Qdrant index và traces
được lưu cục bộ tại `starter/results/cassini-advanced/`.
