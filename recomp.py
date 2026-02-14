import zlib
import lzma

INPUT_FILE = "raw.imgc"
OUTPUT_FILE = "raw.imga"

CHUNK_SIZE = 1024 * 1024


def convert_imgc_to_imga():

    lzma_compressor = lzma.LZMACompressor(
        format=lzma.FORMAT_XZ,
        preset=lzma.PRESET_EXTREME | 9
    )

    with open(INPUT_FILE, "rb") as fin, open(OUTPUT_FILE, "wb") as fout:

        data_buffer = b""
        decompressor = zlib.decompressobj()

        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break

            data_buffer += chunk

            while data_buffer:
                try:
                    decompressed = decompressor.decompress(data_buffer)

                    fout.write(lzma_compressor.compress(decompressed))

                    # Move forward in buffer
                    data_buffer = decompressor.unused_data

                    if decompressor.eof:
                        decompressor = zlib.decompressobj()

                    if not decompressor.unused_data:
                        break

                except zlib.error:
                    # Not enough data yet
                    break

        # Flush remaining decompressed data
        fout.write(lzma_compressor.flush())


if __name__ == "__main__":
    convert_imgc_to_imga()
