function classify()
% CLASSIFY  Module 5 — SVM + kNN Classification (MATLAB)
%
% Loads feature matrix, trains SVM + kNN,
% runs 5-fold cross validation, plots confusion matrices.
%
% Usage:
%   classify()

    BASE_DIR    = 'E:/CTFS_Harmonic_Fingerprinting';
    INSTRUMENTS = {'sitar', 'veena', 'guitar'};
    N_FOLDS     = 5;

    fprintf('MODULE 5 — CLASSIFICATION\n');
    fprintf('%s\n', repmat('=',1,45));

    % ── Load feature matrix ───────────────────────────────────────
    feat_path = fullfile(BASE_DIR, 'data', 'features', 'feature_matrix.csv');
    fprintf('\nLoading feature matrix...\n');

    T = readtable(feat_path);
    feat_cols = startsWith(T.Properties.VariableNames, 'h');
    X = table2array(T(:, feat_cols));
    y = T.instrument;

    fprintf('  X shape : %d x %d\n', size(X,1), size(X,2));

    % ── Scale features ────────────────────────────────────────────
    mu    = mean(X, 1);
    sigma = std(X, 0, 1) + 1e-9;
    X_scaled = (X - mu) ./ sigma;

    % ── Encode labels ─────────────────────────────────────────────
    [y_enc, classes] = grp2idx(y);

    fprintf('\nRunning 5-fold stratified cross-validation...\n');

    % ── SVM ───────────────────────────────────────────────────────
    fprintf('\n  [SVM - RBF Kernel]\n');
    svm_template = templateSVM('KernelFunction', 'rbf', ...
                               'BoxConstraint', 10, ...
                               'KernelScale', 'auto', ...
                               'Standardize', false);
    svm_mdl = fitcecoc(X_scaled, y, 'Learners', svm_template);
    cv_svm  = crossval(svm_mdl, 'KFold', N_FOLDS);
    svm_loss = kfoldLoss(cv_svm);
    svm_acc  = (1 - svm_loss) * 100;
    svm_pred = kfoldPredict(cv_svm);
    fprintf('  Accuracy : %.2f%%\n', svm_acc);

    % ── kNN ───────────────────────────────────────────────────────
    fprintf('\n  [kNN - k=5]\n');
    knn_mdl = fitcknn(X_scaled, y, 'NumNeighbors', 5, ...
                      'Standardize', false);
    cv_knn  = crossval(knn_mdl, 'KFold', N_FOLDS);
    knn_loss = kfoldLoss(cv_knn);
    knn_acc  = (1 - knn_loss) * 100;
    knn_pred = kfoldPredict(cv_knn);
    fprintf('  Accuracy : %.2f%%\n', knn_acc);

    % ── Summary ───────────────────────────────────────────────────
    fprintf('\n%s\n', repmat('=',1,45));
    fprintf('  RESULTS SUMMARY\n');
    fprintf('%s\n', repmat('=',1,45));
    fprintf('  %-15s %10s\n', 'Classifier', 'Accuracy');
    fprintf('  %s\n', repmat('-',1,30));
    fprintf('  %-15s %9.2f%%\n', 'SVM (RBF)',  svm_acc);
    fprintf('  %-15s %9.2f%%\n', 'kNN (k=5)',  knn_acc);

    % ── Confusion matrices ────────────────────────────────────────
    fig_dir = fullfile(BASE_DIR, 'results', 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    plot_cm(y, svm_pred, INSTRUMENTS, 'SVM (RBF)', ...
            fullfile(fig_dir, 'confusion_matrix_SVM.png'));
    plot_cm(y, knn_pred, INSTRUMENTS, 'kNN (k=5)', ...
            fullfile(fig_dir, 'confusion_matrix_kNN.png'));

    % ── Save report CSV ───────────────────────────────────────────
    res_dir = fullfile(BASE_DIR, 'results');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end
    fid = fopen(fullfile(res_dir, 'classification_report.csv'), 'w');
    fprintf(fid, 'classifier,accuracy\n');
    fprintf(fid, 'SVM (RBF),%.2f\n', svm_acc);
    fprintf(fid, 'kNN (k=5),%.2f\n', knn_acc);
    fclose(fid);

    fprintf('\nMODULE 5 COMPLETE\n');
    fprintf('Proceed to Module 6 — ANOVA Raga Effect\n');
end


% ── Helper: plot confusion matrix ────────────────────────────────
function plot_cm(y_true, y_pred, classes, title_str, save_path)
    cm = confusionmat(y_true, y_pred, 'Order', classes);

    figure('Color', [0.1 0.1 0.18], 'Position', [100 100 600 500]);
    imagesc(cm);
    colormap(flipud(hot));
    colorbar;

    ax = gca;
    ax.Color           = [0.1 0.1 0.18];
    ax.XColor          = 'white';
    ax.YColor          = 'white';
    ax.XTick           = 1:numel(classes);
    ax.YTick           = 1:numel(classes);
    ax.XTickLabel      = classes;
    ax.YTickLabel      = classes;
    ax.FontSize        = 12;

    % Annotate cells
    thresh = max(cm(:)) / 2;
    for i = 1:numel(classes)
        for j = 1:numel(classes)
            c = 'white';
            if cm(i,j) > thresh, c = 'black'; end
            text(j, i, num2str(cm(i,j)), ...
                 'HorizontalAlignment', 'center', ...
                 'Color', c, 'FontSize', 13, 'FontWeight', 'bold');
        end
    end

    xlabel('Predicted', 'Color', 'white', 'FontSize', 12);
    ylabel('True',      'Color', 'white', 'FontSize', 12);
    title(['Confusion Matrix — ' title_str], ...
          'Color', 'white', 'FontSize', 13, 'FontWeight', 'bold');

    saveas(gcf, save_path);
    fprintf('  Plot saved: %s\n', save_path);
    close(gcf);
end

