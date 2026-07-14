# 5단계: Google Colab에서 Streamlit 앱 실행하기 (cloudflared 터널)

아래 셀들을 **순서대로, 각각 별도의 Colab 셀**에 붙여넣어 실행하세요.
(app.py 내용을 직접 셀에 붙여넣어 실행하면 안 됩니다 — Streamlit은 파일로 저장한 뒤
`streamlit run`으로 별도 서버 프로세스를 띄우고, 터널로 외부에 노출해야 합니다.)

cloudflared 방식은 localtunnel과 달리 접속 시 비밀번호/IP 입력이 필요 없어 더 간단합니다.
실제로 로컬 환경에서 아래 코드 그대로 서버 실행 → 터널 생성 → 접속까지 전체 플로우를
검증했습니다 (HTTP 200, 페이지 정상 렌더링 확인).

## 셀 1: 라이브러리 설치
```python
!pip install -q streamlit scikit-learn joblib pandas
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```

## 셀 2: 파일 존재 여부 확인
1~4단계 코드를 먼저 실행해서 `risk_model.pkl`, `scored_stocks.csv`, `app.py`가 이미 있어야 합니다.

```python
import os

required_files = ["app.py", "risk_model.pkl", "scored_stocks.csv"]
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    print(f"다음 파일이 없습니다: {missing}")
    print("1~4단계 코드를 먼저 실행해 파일을 생성한 뒤 다시 시도하세요.")
else:
    print("모든 파일이 준비되었습니다:", required_files)
```

## 셀 3: Streamlit 서버 실행 (백그라운드, 8501 포트)
```python
import subprocess
import time

# 기존에 떠 있을 수 있는 프로세스 정리
!pkill -f streamlit
time.sleep(2)

log_file = open("streamlit_log.txt", "w")
streamlit_process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=log_file, stderr=subprocess.STDOUT
)
time.sleep(8)  # Streamlit이 완전히 뜰 때까지 대기

if streamlit_process.poll() is not None:
    print("❌ Streamlit 서버가 시작되지 못했습니다. 에러 로그:")
    with open("streamlit_log.txt") as f:
        print(f.read())
else:
    print("✅ Streamlit 서버 실행 중 (app.py)")
```

## 셀 4: cloudflared 터널 실행 + 접속 URL 탐색
```python
import subprocess
import re
import time

!pkill -f cloudflared
time.sleep(1)

cloudflared_process = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
)

url_found = False
start = time.time()
for line in cloudflared_process.stdout:
    match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
    if match:
        print("접속 링크:", match.group(0))
        print("⏳ 링크가 뜬 후 10~20초 정도 기다렸다가 클릭해주세요 (DNS 전파 시간).")
        url_found = True
        break
    if time.time() - start > 30:
        break

if not url_found:
    print("❌ 터널 링크를 찾지 못했습니다. 셀 5로 로그를 확인하세요.")
```
비밀번호나 IP 입력 없이 링크만 클릭하면 바로 앱 화면이 뜹니다.

## 셀 5: 로그 확인
```python
print("=== Streamlit 로그 ===")
with open("streamlit_log.txt") as f:
    print(f.read())

print("\nStreamlit 프로세스 상태:", "실행 중" if streamlit_process.poll() is None else "종료됨")
print("Cloudflared 프로세스 상태:", "실행 중" if cloudflared_process.poll() is None else "종료됨")
```

## 셀 6: 서버 종료
```python
!pkill -f streamlit
!pkill -f cloudflared
print("Streamlit 서버와 cloudflared 터널을 종료했습니다.")
```

---

### 자주 발생하는 문제
- **`ModuleNotFoundError: No module named 'streamlit'`**: 셀 1을 실행하지 않았거나, app.py
  코드를 파일로 저장하지 않고 직접 셀에서 실행했을 때 발생합니다.
- **터널 링크 접속 시 502/530 에러**: 셀 3의 Streamlit 서버가 아직 뜨는 중이거나 실패했을
  수 있습니다. 셀 5로 로그를 확인한 뒤 셀 3부터 다시 실행하세요.
- **터널 링크를 못 찾음**: 네트워크가 느리면 셀 4의 30초 제한에 걸릴 수 있습니다. 셀 4를
  다시 실행해보세요.
