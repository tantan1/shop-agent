# syntax=docker/dockerfile:1
# 多阶段构建，最终镜像用 scratch（静态二进制，无运行时依赖）
# 目标：镜像 10MB 级别、运行内存 ~5-10MB、零第三方依赖

FROM golang:1.23-alpine AS builder
WORKDIR /build

# 国内 GOPROXY 加速（仅缺依赖时需要；本服务为零依赖，走标准库）
ENV GOPROXY=https://goproxy.cn,direct
ENV CGO_ENABLED=0

COPY go.mod ./
COPY main.go ./
# -trimpath 消除构建路径泄漏；-s -w 剥离符号表/调试信息压缩体积
RUN go build -trimpath -ldflags="-s -w" -o /out/mock-llm .

FROM scratch
COPY --from=builder /out/mock-llm /mock-llm
# 非 root 运行（静态二进制 + scratch 无需用户目录）
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/mock-llm"]