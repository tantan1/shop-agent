# 多阶段构建：Rust 官方镜像编译，最终镜像用轻量 debian-slim
# 默认启用 rsproxy.cn 国内镜像源加速 crates 下载；如需官方源传入 --build-arg CARGO_MIRROR=0
FROM rust:1.94-slim AS builder
WORKDIR /app

# 默认注入国内 cargo crate 镜像源（rsproxy.cn）
ARG CARGO_MIRROR=1
RUN if [ "$CARGO_MIRROR" = "1" ]; then \
      printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n[registries.rsproxy]\nindex = "https://rsproxy.cn/crates.io-index"\n[net]\ngit-fetch-with-cli = true\n' \
        > /usr/local/cargo/config.toml; \
    fi
# sparse 协议拉取索引更快（Rust 1.68+ 默认，这里显式声明以兼容老工具链）
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# 直接拷贝全部源码并编译。注意：不要为做「依赖层缓存」而先用空 main.rs 编译——
# 那样 COPY 覆盖真实源码后 cargo 增量缓存可能不重编 order-service crate，
# 导致最终二进制仍是空 main（进程静默退出码 0、无任何输出）。
COPY . .
# BuildKit cache mount：cargo 注册表缓存 + target 编译产物跨构建复用，不与镜像层绑定
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
# sqlx 的 postgres 驱动是纯 Rust 实现，运行时不依赖 libpq；仅需 ca-certificates（可选）
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/order-service /usr/local/bin/order-service
EXPOSE 8080
CMD ["order-service"]
