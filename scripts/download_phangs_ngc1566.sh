#!/bin/zsh
set -euo pipefail

workspace_dir="/Users/meow/Latex/KT+SpiralArms"
data_dir="${workspace_dir}/data/phangs/NGC1566"
output_dir="${workspace_dir}/outputs/implications/NGC1566"
mkdir -p "${data_dir}" "${output_dir}"

download_product() {
    local product_id="$1"
    local filename="$2"
    local expected_bytes="$3"
    local destination="${data_dir}/${filename}"
    local partial="${destination}.part"
    local url="https://dataportal.eso.org/dataPortal/file/${product_id}"

    if [[ -f "${destination}" ]] && \
       [[ "$(stat -f%z "${destination}")" == "${expected_bytes}" ]]; then
        if [[ ! -s "${destination}.sha256" ]]; then
            shasum -a 256 "${destination}" > "${destination}.sha256"
        fi
        print "Already complete: ${filename}"
        return
    fi

    if [[ -e "${destination}" ]]; then
        print -u2 "Existing file has the wrong size: ${destination}"
        return 1
    fi

    print "Downloading ${filename} (${expected_bytes} bytes)"
    curl --fail --location --retry 4 --retry-delay 10 \
        --output "${partial}" "${url}"

    local received_bytes
    received_bytes="$(stat -f%z "${partial}")"
    if [[ "${received_bytes}" != "${expected_bytes}" ]]; then
        print -u2 "Size check failed for ${filename}: ${received_bytes} bytes"
        return 1
    fi
    mv "${partial}" "${destination}"
    shasum -a 256 "${destination}" > "${destination}.sha256"
    print "Completed: ${destination}"
}

download_product \
    "ADP.2025-05-14T09:12:30.227" \
    "NGC1566_catalog.fits" \
    "1483200"
download_product \
    "ADP.2025-05-14T09:12:30.228" \
    "NGC1566_nebulae_mask.fits" \
    "4268160"
download_product \
    "ADP.2021-07-16T10:20:56.422" \
    "NGC1566_MAPS_copt_0.80asec.fits" \
    "452128320"

print "All NGC 1566 PHANGS products are ready."
