function results = compute_ctfs(instrument, raga)
% COMPUTE_CTFS  Module 2 — CTFS Harmonic Analysis
%
% Loads WAV file, reads pakad annotation, extracts 100ms
% Hann-windowed frames, estimates f0 via autocorrelation,
% computes N=32 CTFS harmonic coefficients per frame.
%
% Usage:
%   results = compute_ctfs('sitar', 'yaman')
%   results = compute_ctfs('guitar', '')
%
% Output:
%   results — struct array with fields:
%       .instrument  : string
%       .raga        : string
%       .frame_idx   : int
%       .f0_hz       : fundamental frequency
%       .magnitudes  : [1 x 32] normalized |cn|
%       .phases      : [1 x 32] angle(cn)

    % ── Parameters ───────────────────────────────────────────────
    N_HARMONICS   = 32;
    FRAME_MS      = 100;
    OVERLAP_PCT   = 0.50;
    TARGET_SR     = 22050;
    ENERGY_THRESH = 0.01;

    BASE_DIR  = 'E:/CTFS_Harmonic_Fingerprinting';
    RAW_DIR   = fullfile(BASE_DIR, 'data', 'raw');
    ANNOT_DIR = fullfile(BASE_DIR, 'data', 'annotations');
    OUT_DIR   = fullfile(BASE_DIR, 'data', 'ctfs');

    if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

    results = struct('instrument', {}, 'raga', {}, 'frame_idx', {}, ...
                     'f0_hz', {}, 'magnitudes', {}, 'phases', {});

    % ── Guitar: no annotation ────────────────────────────────────
    if strcmp(instrument, 'guitar')
        guitar_dir = fullfile(RAW_DIR, 'guitar');
        wav_files  = dir(fullfile(guitar_dir, '*.wav'));

        for f = 1:numel(wav_files)
            fpath    = fullfile(guitar_dir, wav_files(f).name);
            [x, sr]  = audioread(fpath);
            if size(x,2) > 1, x = mean(x,2); end
            x = x / (max(abs(x)) + 1e-9);
            if sr ~= TARGET_SR
                x = resample(x, TARGET_SR, sr);
                sr = TARGET_SR;
            end
            r = extract_frames(x, sr, instrument, 'none', ...
                               wav_files(f).name, N_HARMONICS, ...
                               FRAME_MS, OVERLAP_PCT, ENERGY_THRESH);
            results = [results, r];
        end
        save_results(results, instrument, OUT_DIR);
        return;
    end

    % ── Sitar / Veena: use pakad annotation ──────────────────────
    audio_path = fullfile(RAW_DIR, instrument, [raga '.wav']);
    annot_path = fullfile(ANNOT_DIR, [instrument '_' raga '.json']);

    if ~exist(audio_path, 'file')
        error('Audio not found: %s', audio_path);
    end
    if ~exist(annot_path, 'file')
        error('Annotation not found: %s', annot_path);
    end

    % Read annotation JSON
    fid  = fopen(annot_path);
    raw  = fread(fid, inf, 'uint8=>char')';
    fclose(fid);
    annot = jsondecode(raw);

    pakad_start = annot.pakad.start;
    pakad_end   = annot.pakad.end;
    fprintf('  Pakad region: %.1fs → %.1fs\n', pakad_start, pakad_end);

    % Load only pakad region
    [x, sr] = audioread(audio_path, ...
        [round(pakad_start*TARGET_SR)+1, round(pakad_end*TARGET_SR)]);
    if size(x,2) > 1, x = mean(x,2); end
    x = x / (max(abs(x)) + 1e-9);

    results = extract_frames(x, sr, instrument, raga, ...
                             [raga '.wav'], N_HARMONICS, ...
                             FRAME_MS, OVERLAP_PCT, ENERGY_THRESH);

    save_results(results, instrument, OUT_DIR);
    fprintf('  Done: %d frames extracted\n', numel(results));
end


% ── Frame Extraction ─────────────────────────────────────────────────
function results = extract_frames(x, sr, instrument, raga, filename, ...
                                   N, frame_ms, overlap, thresh)
    frame_len  = round(frame_ms * 1e-3 * sr);
    hop_len    = round(frame_len * (1 - overlap));
    hann_win   = hann(frame_len);
    num_frames = floor((length(x) - frame_len) / hop_len) + 1;

    results = struct('instrument', {}, 'raga', {}, 'frame_idx', {}, ...
                     'f0_hz', {}, 'magnitudes', {}, 'phases', {});
    count = 0;

    for i = 1:num_frames
        start_idx = (i-1)*hop_len + 1;
        frame     = x(start_idx : start_idx + frame_len - 1);

        % Skip silent frames
        rms = sqrt(mean(frame.^2));
        if rms < thresh, continue; end

        % Apply Hann window
        frame_win = frame .* hann_win;

        % Estimate f0
        f0 = estimate_f0(frame_win, sr);
        if isempty(f0) || f0 < 80 || f0 > 1200, continue; end

        % Compute CTFS
        [mags, phases] = ctfs_coefficients(frame_win, sr, f0, N);

        count = count + 1;
        results(count).instrument = instrument;
        results(count).raga       = raga;
        results(count).frame_idx  = i;
        results(count).f0_hz      = f0;
        results(count).magnitudes = mags;
        results(count).phases     = phases;
    end

    fprintf('    %s: %d valid frames\n', filename, count);
end


% ── f0 Estimation via Autocorrelation ────────────────────────────────
function f0 = estimate_f0(frame, sr, f0_min, f0_max)
    if nargin < 3, f0_min = 80;  end
    if nargin < 4, f0_max = 1200; end

    corr    = xcorr(frame, 'normalized');
    corr    = corr(length(frame):end);   % positive lags only

    lag_min = max(1, round(sr / f0_max));
    lag_max = min(length(corr)-1, round(sr / f0_min));

    [~, peak_idx] = max(corr(lag_min:lag_max));
    peak_lag = peak_idx + lag_min - 1;
    f0 = sr / peak_lag;
end


% ── CTFS Coefficient Computation ─────────────────────────────────────
function [magnitudes, phases] = ctfs_coefficients(frame, sr, f0, N)
    n_samples  = length(frame);
    T0         = 1.0 / f0;
    t          = (0:n_samples-1)' / sr;

    magnitudes = zeros(1, N);
    phases     = zeros(1, N);

    for n = 1:N
        omega_n      = 2 * pi * n * f0;
        kernel       = exp(-1j * omega_n * t);
        cn           = trapz(t, frame .* kernel) / T0;
        magnitudes(n) = abs(cn);
        phases(n)     = angle(cn);
    end

    % Normalize by c1
    if magnitudes(1) > 1e-9
        magnitudes = magnitudes / magnitudes(1);
    end
end


% ── Save Results ─────────────────────────────────────────────────────
function save_results(results, instrument, out_dir)
    save_path = fullfile(out_dir, [instrument '_ctfs.mat']);
    save(save_path, 'results');
    fprintf('  Saved → %s\n', save_path);
end

