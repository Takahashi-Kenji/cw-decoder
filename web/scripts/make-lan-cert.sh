#!/usr/bin/env bash
#
# LAN 配信用の自己署名証明書を web/certs/ に作る。
#
# なぜ必要か:
#   ブラウザのマイク取得 (getUserMedia) はセキュアコンテキストでしか動かない。
#   localhost は例外扱いなので同じ PC なら HTTP で足りるが、**別の PC から
#   LAN IP (http://192.168.x.x:5173/) で開くとマイクがブロックされる**。
#   HTTPS にすればブロックされない。認証局の証明書は要らない (自己署名で十分。
#   初回だけブラウザに警告が出るので「詳細設定」→「アクセスする」で通す)。
#
# 使い方 (Git Bash。cw-decorder/web/ で実行):
#
#     bash scripts/make-lan-cert.sh 192.168.0.20 192.168.0.21
#
# 引数はこの PC の LAN IP。`ipconfig` か下記で調べる:
#
#     powershell -c "Get-NetIPAddress -AddressFamily IPv4 | Select IPAddress,InterfaceAlias"
#
# 生成物 (web/certs/) は .gitignore 対象。秘密鍵なのでコミットしないこと。
# 有効期限は 825 日 (ブラウザが受け付ける自己署名証明書の上限に合わせた)。
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p certs

if [ "$#" -eq 0 ]; then
  echo "使い方: bash scripts/make-lan-cert.sh <LAN IP> [<LAN IP> ...]" >&2
  echo "例:     bash scripts/make-lan-cert.sh 192.168.0.20" >&2
  exit 1
fi

{
  cat <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3_req
prompt = no

[dn]
CN = cw-decoder-lan

[v3_req]
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF
  # 引数の IP を IP.2 以降として追記する (IP.1 は 127.0.0.1 で固定)
  i=2
  for ip in "$@"; do
    echo "IP.$i = $ip"
    i=$((i + 1))
  done
} > certs/san.cnf

# MSYS_NO_PATHCONV: Git Bash が -subj 等のパス風文字列を勝手に変換するのを防ぐ
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 825 \
  -keyout certs/lan.key -out certs/lan.crt -config certs/san.cnf 2>/dev/null

echo "生成しました: web/certs/lan.crt / lan.key"
MSYS_NO_PATHCONV=1 openssl x509 -in certs/lan.crt -noout -ext subjectAltName
echo
echo "次: npm run dev:lan で HTTPS + LAN 公開で起動します。"
