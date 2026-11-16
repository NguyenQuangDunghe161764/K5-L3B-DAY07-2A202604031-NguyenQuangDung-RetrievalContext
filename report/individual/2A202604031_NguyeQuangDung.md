# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn: OpenStax Biology 2e, sections 1.1–8.3; 35 section pages; CC BY-NC-SA 4.0. Day 6 reports 1,194 chunks for `A_fixed_500_50` and 750 for `B_paragraph_1000`.
- Vector store: PostgreSQL + pgvector. Embedding theo Day 6 run summary: OpenRouter `openai/text-embedding-3-small`, 1,536 dimensions.
- Day 7 benchmark draft: 15 answerable dev questions from Day 6 and 5 fresh answerable test questions from sections 7–8; the 5 test questions avoid the Day 6 test set. Four draft unanswerable questions add 2 cases per split. Evidence is checked against source pages and fixed-size chunks; full-index absence still needs human verification.
- Agent model: `openai/gpt-4o-mini` via OpenAI API and OpenRouter were each tried for one bounded BM25 smoke case. OpenAI returned HTTP 401; OpenRouter timed out. No valid agent answer was produced.
- Gold evidence and answer grades: Day 6 validates answerable evidence against source pages/chunks; Day 7 unanswerable cases remain candidates until absence is checked against the complete index.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| TF-IDF | top-3, C chunks, filter, budget 800 chars | Lexical baseline | dev | 11/15 (73.3%) | 13/15 (86.7%) | 4/15 (26.7%) | median ~2.4 ms local |
| BM25 | top-3, C chunks, filter, budget 800 chars | Lexical ranking candidate | dev | 14/15 (93.3%) | 15/15 (100%) | 4/15 (26.7%) | median ~2.6 ms local |
| Day 6 dense | top-3, C filter, budget 800 chars; replay saved Day 6 traces | Dense baseline; query time not recorded | dev | 8/15 (53.3%) | 12/15 (80%) | 4/15 (26.7%) | not measured in replay |
| Hybrid RRF | dense top-3 + BM25 top-3, budget 800 chars | Fuse ranks, not raw scores | dev | 13/15 (86.7%) | 14/15 (93.3%) | 4/15 (26.7%) | not measured in replay |
| Cải thiện 1: BM25 | top-3, context 2,000 chars | Increase context budget to include retrieved evidence | dev | 14/15 (93.3%) | 15/15 (100%) | 15/15 (100%) | median ~2.5 ms local |
| Cải thiện 2: Hybrid RRF | dense top-3 + BM25 top-3, context 2,000 chars | Fuse ranks and increase context budget | dev | 13/15 (86.7%) | 14/15 (93.3%) | 14/15 (93.3%) | not measured in replay |
| Candidate for agent | BM25 top-3, C chunks, filter, budget 2,000 chars | Best measured dev retrieval/context; answer quality still untested | dev | 14/15 (93.3%) | 15/15 (100%) | 15/15 (100%) | local only |
| Candidate | Not yet run on new test | Must evaluate after dev freeze | test | pending | pending | pending | pending |
| Baseline | TF-IDF, top-3, budget 800 chars | Same new test questions | test | pending | pending | pending | pending |

Dense results are replayed from the saved Day 6 `C_paragraph_filter` traces; local chunks were rebuilt from the Day 6 corpus. Lexical timings are local only; dense and hybrid live latency and API costs are not measured. Traces/summaries: `starter/results/retrieval-dev*.json`. Two observed dev failures:

- `q13` (enzyme/substrate): BM25 had correct evidence at rank 2, but the rank-1 context block used 996 characters and prevented the 823-character evidence block from fitting under budget 1,200. At budget 2,000, Context Evidence Rate for BM25 rose to 15/15.
- `q15` (DNA location): hybrid ranked the evidence second; the first 1,042-character block left insufficient room for the 898-character evidence block under budget 1,200. At budget 2,000, hybrid context evidence rose to 14/15.

The current dev candidate is BM25 with 2,000-character context, based on retrieval/context only; the agent has not been run, so final selection remains provisional.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
- Candidate: OpenRouter controller with BM25 retrieval, maximum 4 searches, top-3, 2,000-character context. Runner records per-query errors.
- Current status: controller/runner implemented; full dev/test agent run, human citation review, and agent metric aggregation are pending a working provider connection.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
