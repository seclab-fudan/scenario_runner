#!/usr/bin/zsh
change_map () {
    xmllint --shell $1 <<EOF
cd //LogicFile/@filepath
set $2
save
EOF
}
