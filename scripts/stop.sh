#!/usr/bin/env bash
# Stop every PhysLoc run on this machine, from any terminal.
#
#   bash scripts/stop.sh
#
# A render container does not die with the process that started it, so stopping
# `run.sh` or `generate` alone leaves containers rendering. This stops the
# front-ends first, so nothing starts new containers, then kills every PhysLoc
# render container, and reports what is left. Finished jobs are kept: running
# the same command again resumes.
set -u

front_ends() {
  # The [x] trick keeps these patterns from matching this script's own processes.
  pgrep -f "physloc[.]cli generate"
  pgrep -f "scripts/[r]un.sh"
  pgrep -f "scripts/[r]un_fast.sh"
}

pids=$(front_ends | sort -u)
if [ -n "$pids" ]; then
  echo "stopping $(echo "$pids" | wc -l) generate / run.sh process(es)"
  kill $pids 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -z "$(front_ends)" ] && break
    sleep 1
  done
  left=$(front_ends | sort -u)
  if [ -n "$left" ]; then
    echo "still alive after 10 s -- forcing"
    kill -9 $left 2>/dev/null
  fi
fi

# Labelled containers (every container docker/kubric.sh starts), plus any started
# before the label existed, recognised by the worker script they run.
ids=$( { docker ps -q --filter label=physloc=1
         for c in $(docker ps -q); do
           docker inspect -f '{{.Id}} {{range .Args}}{{.}} {{end}}' "$c" \
             | grep -q "physloc/render/" && echo "$c"
         done; } | cut -c1-12 | sort -u)
if [ -n "$ids" ]; then
  echo "killing $(echo "$ids" | wc -l) render container(s)"
  echo "$ids" | xargs docker kill > /dev/null
fi

echo "-- now running: $(front_ends | sort -u | grep -c .) generate/run.sh process(es), $(docker ps -q --filter label=physloc=1 | grep -c .) PhysLoc container(s)"
