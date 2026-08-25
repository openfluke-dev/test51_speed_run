# Build from chaosglue / git root (sibling welvet / tide / webgpu):
#   docker build -f welvet/apps/aai/test51/test51_speed_run/Dockerfile -t test51-speed .
FROM golang:1.22-bookworm AS build

WORKDIR /src
COPY welvet /src/welvet
COPY tide /src/tide
COPY webgpu /src/webgpu

WORKDIR /src/welvet/apps/aai/test51/test51_speed_run
RUN go mod tidy && GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/test51_speed_run .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /out/test51_speed_run /app/test51_speed_run
COPY welvet/apps/aai/test51/test51_speed_run/.env.example /app/.env.example
EXPOSE 5151 8080
ENV SPEED_ADDR=0.0.0.0:5151 \
    SPEED_TIDE_ADDR=0.0.0.0:8080 \
    SPEED_CKPT_ROOT=/app/speed_ckpt \
    SPEED_LAYER=all \
    SPEED_FULL=true \
    SPEED_AUTOSTART=true \
    SPEED_RESUME=true \
    SPEED_WORKERS=4
ENTRYPOINT ["/app/test51_speed_run"]
