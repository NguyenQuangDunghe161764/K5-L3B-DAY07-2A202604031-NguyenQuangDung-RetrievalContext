# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `day6_retriever.py` | Adapter dense retrieval, dùng lại embedder và pgvector store từ Day 6 |
| `chat_controller.py` | Controller gọi OpenAI hoặc OpenRouter |
| `retrieval_methods.py` | BM25, hybrid RRF và cache truy vấn |
| `benchmark_retrieval.py` | So sánh TF-IDF, BM25, dense Day 6 và hybrid trên dev/test |
| `run_agent_benchmark.py` | Chạy agent thật và lưu trace theo split |
| `prepare_day7_benchmark.py` | Tạo benchmark Day 7 từ câu hỏi Day 6 và thêm câu thiếu evidence |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Các thành phần code cho dense, TF-IDF, BM25, hybrid RRF và OpenRouter controller đã có. Kết quả benchmark, đánh giá nội dung/citation và báo cáo vẫn cần chạy, kiểm chứng và chấm thủ công.

## 1. Kết nối agent

`OpenAIController` dùng `OPENAI_API_KEY` từ môi trường hoặc `.env` Day 6, mặc định model `gpt-4o-mini`; có thể chọn OpenRouter bằng `--provider openrouter` và `AGENT_MODEL`. Cần API key OpenAI Platform; ChatGPT subscription không thay thế API key. Gọi model thật có thể phát sinh chi phí. Ví dụ chạy agent xem mục 6. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Khung agent chỉ kiểm tra citation ID có trong context; bạn phải đọc để xác nhận nguồn hỗ trợ câu trả lời.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF và BM25 là lexical retrieval; hybrid dùng Reciprocal Rank Fusion, không cộng trực tiếp score khác thang đo.

## 3. Kết nối trực tiếp vector store Day 6

`day6_retriever.py` tái sử dụng `MyEmbedder` / `GeminiEmbedder` và `PgVectorStore` từ Day 6. Adapter đọc `.env` trong thư mục Day 6, truy vấn dimension từ strategy đã index, rồi chỉ đọc dữ liệu khi tìm kiếm; không tạo lại embedding hoặc ghi vào database. Cần cài dependencies Day 6 (`psycopg[binary]`, `python-dotenv`) và cấu hình database/provider theo Day 6. Không đưa key vào lệnh hoặc commit `.env`.

Đặt đoạn kết nối dưới đây trong script chạy từ `starter/`:

```python
from pathlib import Path

from agentic import run_agent
from day6_retriever import connect_day6

retriever = connect_day6(
    Path(r"D:\path\to\Day06\project"),
    strategy="A_fixed_500_50",
)
result = run_agent(question, controller, retriever)
```

`controller` vẫn cần kết nối model thật theo giao thức trong `agentic.py`. Có thể chọn `B_paragraph_1000` hoặc `C_paragraph_filter`; truyền `filters` vào `run_agent` để áp dụng metadata filter. Embedding model, dimension và strategy phải khớp index hiện có. Kết nối lỗi hoặc strategy chưa được index sẽ báo lỗi, không chuyển sang dữ liệu giả.

## 4. Tạo benchmark và so sánh retrieval

Tạo 20 câu retrieval có đáp án và 24 agent cases (15/5 có đáp án, 2/2 thiếu evidence) từ bộ câu hỏi Day 6:

```powershell
python starter\prepare_day7_benchmark.py --questions "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion\starter\data\questions.json" --pages "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion\starter\data\pages.jsonl" --output starter\data\day6_benchmark
```

`questions.json` dùng 15 câu dev Day 6 đã có và 5 câu test mới từ section 7–8; script kiểm tra evidence tồn tại trên trang nguồn và trong chunk 500 ký tự. Các câu test mới không trùng bộ test Day 6 đã dùng khi chọn cấu hình. `cases.json` chỉ chứa câu hỏi và nhãn answerable, không đưa gold/đáp án vào agent. Bốn câu thiếu evidence là bản nháp; trước khi chấm, cần xác nhận absence_reason và kiểm tra toàn bộ index, không chỉ top-k.

So sánh bốn phương pháp trên cùng corpus, split và cấu hình:

```powershell
python starter\benchmark_retrieval.py --questions starter\data\day6_benchmark\questions.json --split dev --day6-project "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion" --strategy A_fixed_500_50 --output starter\results\retrieval-dev.json
```

Runner báo Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian và lỗi riêng; lưu ranking/context để kiểm tra. Dense/hybrid gọi embedding API, nên dùng dev để chọn cấu hình trước khi chạy test. Chỉ chạy test sau khi đã chốt cấu hình; phương pháp/cấu hình cần cùng bộ câu hỏi và index. TF-IDF, BM25 và hybrid dùng các chunk đã index trong PostgreSQL.

Nếu PostgreSQL đang dừng, có thể chạy lexical và hybrid so sánh với top-3 dense đã lưu từ Day 6:

```powershell
python starter\benchmark_retrieval.py --questions starter\data\day6_benchmark\questions.json --split dev --day6-project "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion" --strategy C_paragraph_filter --offline-day6-traces "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion\starter\results\traces-20261114T053620Z.json" --output starter\results\retrieval-dev.json
```

Chế độ này tái tạo chunk Day 6 từ corpus cục bộ và replay kết quả dense đã lưu; chỉ dùng được với query/filter khớp trace và cutoff dense top-3. Nó không thay cho truy vấn database live hoặc đánh giá trên test mới.

## 5. Chuyển dữ liệu Day 6 để chạy baseline

Để chạy baseline TF-IDF offline, cần xuất chunk và câu hỏi thành hai JSON array. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 6. Chạy agent thật và tổng hợp đánh giá

Thử một câu dev với OpenAI API và BM25 trên corpus Day 6 (tối đa hai lượt gọi chat):

```powershell
python starter\run_agent_benchmark.py --cases starter\data\day6_benchmark\cases.json --split dev --day6-project "D:\Day06\K5-L3B-DAY06-NguyenQuangDung-2A202604031-DataIngestion" --strategy C_paragraph_filter --retriever bm25 --budget 2000 --provider openai --model gpt-4o-mini --max-searches 1 --limit 1 --output starter\results\agent-smoke-dev.json
```

Thay model nếu muốn; không đặt key trong lệnh. BM25 dùng corpus Day 6 tại chỗ, không cần PostgreSQL; controller gọi OpenAI API có thể tính phí. Nếu smoke test thành công, bỏ `--limit 1` và tăng `--max-searches` để chạy toàn bộ dev. Để dùng dense, chọn `--retriever dense` và cần database Day 6 đang chạy; hybrid cần database và corpus tại chỗ. Chỉ chạy test sau khi đã chọn cấu hình trên dev. Runner không gửi nhãn, `absence_reason` hay gold cho model; output là JSON array traces tương thích với `agent_metrics.py`, lỗi mỗi query được lưu thành status `error`. Ghi chi phí API từ usage của provider.

Đọc trace rồi tự tạo ba JSON array để tổng hợp nhãn:

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
