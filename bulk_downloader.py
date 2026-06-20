from binance_bulk_downloader.downloader import BinanceBulkDownloader
# Download 15m, 1h, and 4h data for XRPUSDT
for interval in ['15m', '1h', '4h']:
    downloader = BinanceBulkDownloader(
        data_type="klines",
        data_frequency=interval,
        asset="spot",
        symbols=["XRPUSDT"]
    )
    downloader.run_download()