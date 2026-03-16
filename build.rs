fn main() {
    #[cfg(feature = "grpc")]
    {
        tonic_build::compile_protos("proto/axon_infer.proto")
            .expect("failed to compile axon_infer.proto");
    }
}
