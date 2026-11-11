"""GitHub가 1시간마다 실행하는 계산 스크립트. (PC의 paper_status.py와 같은 계산)

runs.json의 모의매매를 '시작일부터 지금까지' 다시 계산해 results.json으로 저장한다.
핸드폰 페이지는 results.json만 읽어서 보여 준다(핸드폰이 업비트에 직접 요청하지 않음).

예) python update_results.py runs.json results.json
"""
import json
import sys
from datetime import datetime, timedelta, timezone

from autotrade.forward import compute_runs


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    results = compute_runs(data)
    now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"updated": now, "results": results}, f, ensure_ascii=False, default=str)
    ok = sum(r.get("status") != "error" for r in results)
    print(f"{now} 계산 완료: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
