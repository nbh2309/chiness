#!/usr/bin/env bash
# Tải Pikafish (engine cờ tướng mạnh nhất, mã nguồn mở) vào server/engines/.
# Cần: curl, 7z (apt install p7zip-full), python3.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p engines && cd engines

url=$(curl -fsSL https://api.github.com/repos/official-pikafish/Pikafish/releases/latest |
      python3 -c 'import json,sys; a=json.load(sys.stdin)["assets"]; print(next(x["browser_download_url"] for x in a if x["name"].endswith(".7z")))')
echo "Tải $url"
curl -fL -o pikafish.7z "$url"
rm -rf pikafish_release && mkdir pikafish_release
7z x -opikafish_release pikafish.7z >/dev/null

# Chọn bản build hợp với CPU (nhanh nhất mà CPU hỗ trợ)
flags=$(grep -m1 flags /proc/cpuinfo || true)
pick=""
for variant in avx512 vnni bmi2 avx2 sse41-popcnt ssse3 x86-64; do
  case $variant in
    avx512) grep -q avx512bw <<<"$flags" || continue ;;
    vnni) grep -q avx512_vnni <<<"$flags" || continue ;;
    bmi2) grep -q bmi2 <<<"$flags" || continue ;;
    avx2) grep -q avx2 <<<"$flags" || continue ;;
    sse41-popcnt) grep -q sse4_1 <<<"$flags" || continue ;;
  esac
  pick=$(find pikafish_release -ipath '*linux*' -type f -iname "*${variant}*" | head -n1)
  [ -n "$pick" ] && break
done
[ -n "$pick" ] || pick=$(find pikafish_release -ipath '*linux*' -type f -iname 'pikafish*' ! -iname '*.nnue' | head -n1)
[ -n "$pick" ] || { echo "Không tìm thấy bản Linux trong gói tải về"; exit 1; }
cp "$pick" pikafish && chmod +x pikafish
nnue=$(find pikafish_release -type f -name '*.nnue' | head -n1)
[ -n "$nnue" ] && cp "$nnue" pikafish.nnue
rm -rf pikafish_release pikafish.7z
echo "Đã cài: $(pwd)/pikafish ($(basename "$pick"))"
printf 'uci\nquit\n' | ./pikafish | grep -m1 'id name' || true
