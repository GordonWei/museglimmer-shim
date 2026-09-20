#!/bin/bash
# Management wrapper around launchd for the Muse Glimmer shim.
# Usage: ./museglimmer-shim.sh {start|stop|restart|status|logs}
set -euo pipefail

LABEL="com.wei.museglimmer-shim"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_TEMPLATE="$DIR/$LABEL.plist.template"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_N="$(id -u)"
DOMAIN="gui/$UID_N"

# The plist is generated (not committed) so the repo isn't tied to one
# machine's absolute path — __DIR__ in the template is substituted with
# wherever this checkout actually lives.
install_plist() {
    sed "s#__DIR__#$DIR#g" "$PLIST_TEMPLATE" > "$PLIST_DST"
}

case "${1:-}" in
    start)
        install_plist
        launchctl bootstrap "$DOMAIN" "$PLIST_DST" 2>/dev/null || launchctl kickstart -k "$DOMAIN/$LABEL"
        echo "started. tail -f logs/stdout.log to watch model loading (first load can take a minute+)."
        ;;
    stop)
        launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || echo "(already stopped)"
        ;;
    restart)
        launchctl kickstart -k "$DOMAIN/$LABEL" 2>/dev/null || { install_plist; launchctl bootstrap "$DOMAIN" "$PLIST_DST"; }
        ;;
    status)
        launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|pid" || echo "not loaded"
        echo "---"
        curl -s -m 3 http://127.0.0.1:8091/health || echo "(health check failed — still loading, or not running)"
        ;;
    logs)
        tail -n 100 -f "$DIR/logs/stdout.log" "$DIR/logs/stderr.log"
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
