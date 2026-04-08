function anova_raga_effect()
% ANOVA_RAGA_EFFECT  Module 6 — ANOVA Raga Effect Analysis (MATLAB)
%
% Tests whether raga context significantly shifts the
% harmonic fingerprint of the same instrument.
%
% Usage:
%   anova_raga_effect()

    BASE_DIR    = 'E:/CTFS_Harmonic_Fingerprinting';
    N_HARMONICS = 32;
    RAGA_MAP    = struct('sitar', {{'yaman','bhairavi'}}, ...
                         'veena', {{'bhairavi','desh'}});
    instruments = fieldnames(RAGA_MAP);

    fprintf('MODULE 6 — ANOVA RAGA EFFECT ANALYSIS\n');
    fprintf('%s\n', repmat('=',1,45));

    fig_dir = fullfile(BASE_DIR, 'results', 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    res_dir = fullfile(BASE_DIR, 'results');
    fid_csv = fopen(fullfile(res_dir, 'anova_results.csv'), 'w');
    fprintf(fid_csv, 'instrument,mean_F,min_p,sig_harmonics,significant\n');

    for k = 1:numel(instruments)
        inst  = instruments{k};
        ragas = RAGA_MAP.(inst);

        fprintf('\n[%s] — Ragas: %s\n', upper(inst), strjoin(ragas,', '));

        % ── Load CTFS .mat ────────────────────────────────────────
        mat_path = fullfile(BASE_DIR, 'data', 'segments', inst, ...
                            [inst '_ctfs.mat']);
        if ~exist(mat_path, 'file')
            fprintf('  .mat not found: %s\n', mat_path);
            continue;
        end
        D = load(mat_path);

        % ── Group by raga ─────────────────────────────────────────
        raga_data = struct();
        for r = 1:numel(ragas)
            raga = ragas{r};
            idx  = strcmp(D.raga_labels, raga);
            raga_data.(raga) = D.all_mags(idx, :);
            fprintf('  %s: %d frames\n', raga, sum(idx));
        end

        % ── ANOVA per harmonic ────────────────────────────────────
        f_stats = zeros(1, N_HARMONICS);
        p_vals  = zeros(1, N_HARMONICS);

        for n = 1:N_HARMONICS
            groups = cell(1, numel(ragas));
            for r = 1:numel(ragas)
                groups{r} = raga_data.(ragas{r})(:, n);
            end
            [p, tbl] = anova1(cell2mat(cellfun(@(x) x(:), ...
                              groups, 'UniformOutput', false)), ...
                              [], 'off');
            f_stats(n) = tbl{2,5};
            p_vals(n)  = p;
        end

        sig_n  = sum(p_vals < 0.05);
        mean_f = mean(f_stats);
        min_p  = min(p_vals);

        fprintf('  Significant harmonics: %d/%d\n', sig_n, N_HARMONICS);
        fprintf('  Mean F-statistic     : %.2f\n', mean_f);
        fprintf('  Min p-value          : %.6f\n', min_p);
        if min_p < 0.05
            fprintf('  Raga SIGNIFICANTLY affects harmonic fingerprint\n');
            sig_str = 'YES';
        else
            fprintf('  No significant raga effect\n');
            sig_str = 'NO';
        end

        fprintf(fid_csv, '%s,%.2f,%.6f,%d,%s\n', ...
                inst, mean_f, min_p, sig_n, sig_str);

        % ── Plot ─────────────────────────────────────────────────
        raga_colors = {'#B85042', '#A7BEAE', '#E7E8D1'};
        bg = [0.1 0.1 0.18];

        figure('Color', bg, 'Position', [50 50 1200 480]);

        % Left: harmonic profiles per raga
        subplot(1,2,1);
        set(gca, 'Color', bg, 'XColor','w', 'YColor','w');
        hold on;
        for r = 1:numel(ragas)
            raga  = ragas{r};
            mags  = raga_data.(raga);
            mu    = mean(mags, 1);
            sg    = std(mags, 0, 1);
            c     = sscanf(raga_colors{r}(2:end),'%2x%2x%2x',[1 3])/255;
            plot(1:N_HARMONICS, mu, 'Color', c, 'LineWidth', 2.5, ...
                 'DisplayName', ['Raga ' raga]);
            fill([1:N_HARMONICS, N_HARMONICS:-1:1], ...
                 [mu+sg, fliplr(mu-sg)], c, ...
                 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                 'HandleVisibility', 'off');
        end
        xlabel('Harmonic Index (n)', 'Color','w','FontSize',11);
        ylabel('Normalized |cₙ|',   'Color','w','FontSize',11);
        title([upper(inst(1)) inst(2:end) ' — Profiles by Raga'], ...
              'Color','w','FontSize',12,'FontWeight','bold');
        legend('TextColor','w','Color',[0.15 0.15 0.25],'FontSize',10);
        grid on; grid minor;
        set(gca,'GridColor','w','GridAlpha',0.12,'MinorGridAlpha',0.06);

        % Right: F-statistic bar chart
        subplot(1,2,2);
        set(gca, 'Color', bg, 'XColor','w', 'YColor','w');
        bar_cols = zeros(N_HARMONICS, 3);
        for n = 1:N_HARMONICS
            if p_vals(n) < 0.05
                bar_cols(n,:) = [0.72 0.31 0.26];   % red
            else
                bar_cols(n,:) = [0.33 0.33 0.47];   % grey
            end
        end
        hold on;
        for n = 1:N_HARMONICS
            bar(n, f_stats(n), 'FaceColor', bar_cols(n,:), ...
                'EdgeColor', 'none', 'FaceAlpha', 0.85);
        end
        yline(3.84, '--w', 'LineWidth', 1.5, ...
              'DisplayName', 'F critical (p=0.05)');
        xlabel('Harmonic Index (n)', 'Color','w','FontSize',11);
        ylabel('F-statistic',        'Color','w','FontSize',11);
        title([upper(inst(1)) inst(2:end) ' — ANOVA F-statistic'], ...
              'Color','w','FontSize',12,'FontWeight','bold');
        legend('TextColor','w','Color',[0.15 0.15 0.25],'FontSize',10);
        grid on;
        set(gca,'GridColor','w','GridAlpha',0.12);

        sgtitle(['Raga Effect Analysis — ' upper(inst(1)) inst(2:end)], ...
                'Color','w','FontSize',14,'FontWeight','bold');

        out_path = fullfile(fig_dir, ['anova_' inst '.png']);
        saveas(gcf, out_path);
        fprintf('  Plot saved: %s\n', out_path);
        close(gcf);
    end

    fclose(fid_csv);

    fprintf('\n%s\n', repmat('=',1,45));
    fprintf('MODULE 6 COMPLETE\n');
    fprintf('Proceed to Module 7 — Visualization\n');
end
