#!/bin/zsh
set -u
set -o pipefail

workspace_dir="/Users/meow/Latex/KT+SpiralArms"
data_root="${workspace_dir}/data/phangs"
output_root="${workspace_dir}/outputs/implications"
log_path="${output_root}/candidate_sample_pipeline.log"
status_path="${output_root}/candidate_sample_status.tsv"
mkdir -p "${data_root}" "${output_root}"

exec > >(tee -a "${log_path}") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

record_status() {
    local galaxy="$1"
    local stage="$2"
    local state="$3"
    print -r -- "$(timestamp)"$'\t'"${galaxy}"$'\t'"${stage}"$'\t'"${state}" >> "${status_path}"
}

download_product() {
    local galaxy="$1"
    local product_id="$2"
    local filename="$3"
    local expected_bytes="$4"
    local data_dir
    if [[ "${galaxy}" == "NGC628" ]]; then
        data_dir="${data_root}"
    else
        data_dir="${data_root}/${galaxy}"
    fi
    local destination="${data_dir}/${filename}"
    local partial="${destination}.part"
    local url="https://dataportal.eso.org/dataPortal/file/${product_id}"
    mkdir -p "${data_dir}" "${output_root}/${galaxy}"

    if [[ -f "${destination}" ]] && \
       [[ "$(stat -f%z "${destination}")" == "${expected_bytes}" ]]; then
        if [[ ! -s "${destination}.sha256" ]]; then
            shasum -a 256 "${destination}" > "${destination}.sha256"
        fi
        print "[$(timestamp)] Already complete: ${filename}"
        return 0
    fi

    if [[ -e "${destination}" ]]; then
        print -u2 "[$(timestamp)] Existing file has the wrong size: ${destination}"
        return 1
    fi

    print "[$(timestamp)] Downloading ${filename} (${expected_bytes} bytes)"
    if ! curl --fail --location --retry 4 --retry-delay 10 \
        --progress-bar --output "${partial}" "${url}"; then
        print -u2 "[$(timestamp)] Download failed: ${filename}"
        return 1
    fi

    local received_bytes
    received_bytes="$(stat -f%z "${partial}")"
    if [[ "${received_bytes}" != "${expected_bytes}" ]]; then
        print -u2 "[$(timestamp)] Size check failed for ${filename}: ${received_bytes} bytes"
        return 1
    fi
    mv "${partial}" "${destination}"
    shasum -a 256 "${destination}" > "${destination}.sha256"
    print "[$(timestamp)] Completed: ${destination}"
}

download_galaxy() {
    local galaxy="$1"
    shift
    record_status "${galaxy}" download started
    local failed=0
    while (( $# >= 3 )); do
        download_product "${galaxy}" "$1" "$2" "$3" || failed=1
        shift 3
    done
    if (( failed )); then
        record_status "${galaxy}" download failed
        return 1
    fi
    record_status "${galaxy}" download complete
}

download_galaxy NGC628 \
    ADP.2025-05-14T09:12:30.213 NGC0628_catalog.fits 1759680 \
    ADP.2025-05-14T09:12:30.214 NGC0628_nebulae_mask.fits 8058240 \
    ADP.2021-07-16T10:20:56.338 NGC0628_MAPS_copt_0.92asec.fits 854029440 \
    ADP.2021-07-16T10:20:56.336 NGC0628_PHANGS_IMAGE_SDSS_r_copt_0.92asec.fits 18161280

download_galaxy NGC1566 \
    ADP.2025-05-14T09:12:30.227 NGC1566_catalog.fits 1483200 \
    ADP.2025-05-14T09:12:30.228 NGC1566_nebulae_mask.fits 4268160 \
    ADP.2021-07-16T10:20:56.422 NGC1566_MAPS_copt_0.80asec.fits 452128320 \
    ADP.2021-07-16T10:20:56.420 NGC1566_PHANGS_IMAGE_SDSS_r_copt_0.80asec.fits 9630720

download_galaxy NGC1300 \
    ADP.2025-05-14T09:12:30.217 NGC1300_catalog.fits 927360 \
    ADP.2025-05-14T09:12:30.218 NGC1300_nebulae_mask.fits 5325120 \
    ADP.2021-07-16T10:20:56.362 NGC1300_MAPS_copt_0.89asec.fits 564168960 \
    ADP.2021-07-16T10:20:56.360 NGC1300_PHANGS_IMAGE_SDSS_r_copt_0.89asec.fits 12012480

download_galaxy NGC4254 \
    ADP.2025-05-14T09:12:30.237 NGC4254_catalog.fits 1823040 \
    ADP.2025-05-14T09:12:30.238 NGC4254_nebulae_mask.fits 4821120 \
    ADP.2021-07-16T10:20:56.482 NGC4254_MAPS_copt_0.89asec.fits 510897600 \
    ADP.2021-07-16T10:20:56.480 NGC4254_PHANGS_IMAGE_SDSS_r_copt_0.89asec.fits 10880640

download_galaxy NGC4303 \
    ADP.2025-05-14T09:12:30.239 NGC4303_catalog.fits 1886400 \
    ADP.2025-05-14T09:12:30.240 NGC4303_nebulae_mask.fits 3237120 \
    ADP.2021-07-16T10:20:56.494 NGC4303_MAPS_copt_0.78asec.fits 342838080 \
    ADP.2021-07-16T10:20:56.492 NGC4303_PHANGS_IMAGE_SDSS_r_copt_0.78asec.fits 7312320

download_galaxy NGC4321 \
    ADP.2025-05-14T09:12:30.241 NGC4321_catalog.fits 1149120 \
    ADP.2025-05-14T09:12:30.242 NGC4321_nebulae_mask.fits 4924800 \
    ADP.2021-07-16T10:20:56.506 NGC4321_MAPS_copt_1.16asec.fits 521732160 \
    ADP.2021-07-16T10:20:56.504 NGC4321_PHANGS_IMAGE_SDSS_r_copt_1.16asec.fits 11111040

download_galaxy NGC4535 \
    ADP.2025-05-14T09:12:30.243 NGC4535_catalog.fits 1203840 \
    ADP.2025-05-14T09:12:30.244 NGC4535_nebulae_mask.fits 2194560 \
    ADP.2021-07-16T10:20:56.518 NGC4535_MAPS_copt_0.56asec.fits 232479360 \
    ADP.2021-07-16T10:20:56.516 NGC4535_PHANGS_IMAGE_SDSS_r_copt_0.56asec.fits 4965120

if [[ "${PHANGS_DOWNLOAD_ONLY:-0}" == "1" ]]; then
    print "[$(timestamp)] Download-only sample pipeline finished."
    exit 0
fi

for galaxy in NGC628 NGC1566 NGC1300 NGC4254 NGC4303 NGC4321 NGC4535; do
    notebook="${workspace_dir}/${galaxy}_PHANGS_implications.ipynb"
    record_status "${galaxy}" analysis started
    print "[$(timestamp)] Executing ${notebook}"
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        "${notebook}"; then
        record_status "${galaxy}" analysis complete
        print "[$(timestamp)] Analysis complete: ${galaxy}"
    else
        record_status "${galaxy}" analysis failed
        print -u2 "[$(timestamp)] Analysis failed: ${galaxy}; continuing with next target."
    fi
done

print "[$(timestamp)] Candidate-sample pipeline finished."
