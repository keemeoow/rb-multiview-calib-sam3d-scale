#!/usr/bin/env bash
# sam-3d-objects 체크포인트를 curl(IPv4 고정)로 직접 다운로드.
# 이 박스에서 `hf download`가 멈추는(IPv6 죽음 + huggingface_hub 1.13.0 연결복구 버그) 문제 우회용.
# - curl -4: IPv4 강제 (DNS가 IPv6 먼저 주는데 IPv6 경로가 죽어 있음 → /etc/gai.conf IPv4 우선도 함께 권장)
# - -C -   : 이어받기 (기존 파일 크기부터)
# - --speed-limit/--speed-time: 15초간 50KB/s 미만이면 abort → 바깥 until이 재시도
# repro/에 두지만 third_party/sam-3d-objects/ 안에 받는다 (아래 cd가 그쪽으로 이동).
set -u
cd "$(dirname "$(readlink -f "$0")")/../third_party/sam-3d-objects" || {
  echo "cd 실패 — repo 안의 repro/에서 실행하세요"; exit 1; }

REPO=facebook/sam-3d-objects
BASE=https://huggingface.co
DEST=checkpoints/hf-download
TOK=$(hf auth token 2>/dev/null); [ -z "$TOK" ] && TOK=$(cat ~/.cache/huggingface/token 2>/dev/null)
[ -z "$TOK" ] && { echo "HF 토큰 없음 — 'hf auth login' 먼저"; exit 1; }
AUTH="Authorization: Bearer $TOK"

echo ">> [$(date '+%T')] 파일 목록/크기 조회 (IPv4)..."
LIST=$(curl -4 -sL -H "$AUTH" "$BASE/api/models/$REPO/tree/main?recursive=true" \
  | python3 -c "import sys,json
d=json.load(sys.stdin)
for x in d:
    if x.get('type')=='file': print(f\"{x['size']}\t{x['path']}\")")
N=$(echo "$LIST" | grep -c .)
echo ">> 총 $N개 파일"
[ "$N" -lt 1 ] && { echo "목록 조회 실패 (토큰/네트워크 확인)"; exit 1; }

echo "$LIST" | while IFS=$'\t' read -r size path; do
  [ -z "$path" ] && continue
  out="$DEST/$path"; mkdir -p "$(dirname "$out")"
  cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$cur" -ge "$size" ] 2>/dev/null; then
    echo "✅ [$(date '+%T')] skip(완료): $path"
    continue
  fi
  echo ">> [$(date '+%T')] 받기: $path  ($(( size/1048576 ))MB, 시작 $(( cur/1048576 ))MB)"
  until curl -4 -L -H "$AUTH" -C - \
        --speed-limit 51200 --speed-time 15 --connect-timeout 20 \
        --progress-bar "$BASE/$REPO/resolve/main/$path" -o "$out"; do
    rc=$?
    cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$size" ] 2>/dev/null; then break; fi
    echo "   ...끊김(rc=$rc, $(( cur/1048576 ))MB) → 2초 후 이어받기"
    sleep 2
  done
done

echo ">> [$(date '+%T')] ✅ 전체 완료 — mv checkpoints/hf-download/checkpoints checkpoints/hf 로 정리"
