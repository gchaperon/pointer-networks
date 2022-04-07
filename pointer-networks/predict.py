import ptrnets


def main():
    dset = ptrnets.TSP("data", 10, "test", "optimal")
    model = ptrnets.PointerNetworkForTSP.load_from_checkpoint(
        "lightning_logs/version_9/checkpoints/epoch=8-step=70316.ckpt"
    )

    points, point_path, indices = dset[0]
    decoded = model.decode(points)
    breakpoint()
    pass


if __name__ == "__main__":
    main()
