function PedestrianTrackingYOLOv4_GPU()
    % Limpieza inicial
    clear all; clc;

    %% ================= GROUND TRUTH =================
    gt = readmatrix('gt.txt');   % frame,id,x,y,w,h,...
    allIoUs  = [];
    allDice  = [];
    frame_number = 0;

    %% --- CONFIGURACIÓN GPU (SEGURA) ---
    useGPU = false;
    if gpuDeviceCount > 0
        try
            parallel.gpu.enableCUDAForwardCompatibility(true);
            gpuDevice();
            useGPU = true;
            fprintf('GPU activada para YOLOv4\n');
        catch
            fprintf('GPU detectada pero no usable, usando CPU\n');
        end
    else
        fprintf('No se detectó GPU, usando CPU\n');
    end

    %% --- CONFIGURACIÓN DE YOLOv4 ---
    name = 'csp-darknet53-coco';
    detectorYOLO = yolov4ObjectDetector(name);

    %% --- CONFIGURACIÓN DE VIDEO ---
    videoFile = 'PETS09-S2L1-raw.webm';   % PETS09-S2L1
    obj.reader = VideoReader(videoFile);
    obj.videoPlayer = vision.VideoPlayer('Position', [29, 597, 643, 386]);

    %% --- TABLA DE ESCALAS ---
    ld = load('pedScaleTable.mat', 'pedScaleTable');
    obj.pedScaleTable = ld.pedScaleTable;

    %% --- INICIALIZACIÓN DE TRACKS ---
    tracks = struct( ...
        'id', {}, 'color', {}, 'bboxes', {}, 'scores', {}, ...
        'kalmanFilter', {}, 'age', {}, 'totalVisibleCount', {}, ...
        'confidence', {}, 'predPosition', {});
    nextId = 1;

    %% --- PARÁMETROS ---
    option.ROI                  = [40 95 400 140];
    option.scThresh             = 0.3;
    option.gatingThresh         = 0.9;
    option.costOfNonAssignment  = 10;
    option.timeWindowSize       = 16;
    option.confidenceThresh     = 0.35;
    option.ageThresh            = 3;
    option.visThresh            = 0.6;

    %% ================= BUCLE PRINCIPAL =================
    while hasFrame(obj.reader)
        frame = readFrame(obj.reader);
        frame_number = frame_number + 1;

        % 1. DETECCIÓN YOLOv4
        [centroids, bboxes, scores] = detectPeopleYOLO(frame);

        %% ========== IoU + DICE ==========
        idxGT = gt(:,1) == frame_number;
        gtBBoxes = gt(idxGT, 3:6);   % [x y w h]

        if ~isempty(bboxes) && ~isempty(gtBBoxes)
            for d = 1:size(bboxes,1)

                detBox = bboxes(d,:);

                % --- IoU ---
                ious = bboxOverlapRatio(detBox, gtBBoxes);
                maxIoU = max(ious);
                allIoUs(end+1) = maxIoU;

                % --- Dice ---
                maskDet = zeros(size(frame,1), size(frame,2));
                maskGT  = zeros(size(frame,1), size(frame,2));

                x = max(1, round(detBox(1)));
                y = max(1, round(detBox(2)));
                w = round(detBox(3));
                h = round(detBox(4));
                maskDet(y:min(y+h,end), x:min(x+w,end)) = 1;

                [~, gtIdx] = max(ious);
                gtBox = gtBBoxes(gtIdx,:);

                xg = max(1, round(gtBox(1)));
                yg = max(1, round(gtBox(2)));
                wg = round(gtBox(3));
                hg = round(gtBox(4));
                maskGT(yg:min(yg+hg,end), xg:min(xg+wg,end)) = 1;

                diceScore = dice(maskDet, maskGT);
                allDice(end+1) = diceScore;
            end
        end
        %% =================================

        % 2. PREDICCIÓN KALMAN
        for i = 1:length(tracks)
            bbox = tracks(i).bboxes(end, :);
            predictedCentroid = predict(tracks(i).kalmanFilter);
            tracks(i).predPosition = ...
                [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end

        % 3. ASIGNACIÓN
        if isempty(tracks) || isempty(bboxes)
            assignments = zeros(0,2);
            unassignedTracks = 1:length(tracks);
            unassignedDetections = 1:size(bboxes,1);
        else
            predBboxes = reshape([tracks(:).predPosition], 4, [])';
            cost = 1 - bboxOverlapRatio(predBboxes, bboxes);
            cost(cost > option.gatingThresh) = 1 + 30;
            [assignments, unassignedTracks, unassignedDetections] = ...
                assignDetectionsToTracks(double(cost), ...
                                         double(option.costOfNonAssignment));
        end

        % 4. ACTUALIZAR TRACKS
        updateTracks();

        % 5. BORRAR / CREAR
        deleteLostTracks();
        createNewTracks();

        % 6. VISUALIZACIÓN
        displayTrackingResults();

        if ~isOpen(obj.videoPlayer), break; end
    end

    fprintf('\nRESULTADOS FINALES:\n');
    fprintf('IoU medio  = %.3f\n', mean(allIoUs));
    fprintf('Dice medio = %.3f\n', mean(allDice));

    %% ================= FUNCIONES =================

    function [centroids, bboxes, scores] = detectPeopleYOLO(img)
        if useGPU
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, ...
                'Threshold', 0.45, ...
                'ExecutionEnvironment', 'gpu');
        else
            [allBboxes, allScores, allLabels] = detect( ...
                detectorYOLO, img, ...
                'Threshold', 0.45);
        end

        idx = (allLabels == 'person');
        bboxes = allBboxes(idx, :);
        scores = allScores(idx, :);

        if ~isempty(bboxes)
            yfoot = min(length(obj.pedScaleTable), ...
                        round(bboxes(:,2) + bboxes(:,4)));
            estHeight = obj.pedScaleTable(yfoot);
            invalid = abs(estHeight - bboxes(:,4)) > ...
                      estHeight * option.scThresh;
            bboxes(invalid, :) = [];
            scores(invalid, :) = [];
        end

        if isempty(bboxes)
            centroids = [];
        else
            centroids = [ ...
                bboxes(:,1) + bboxes(:,3)/2, ...
                bboxes(:,2) + bboxes(:,4)/2 ];
        end
    end

    function updateTracks()
        for i = 1:size(assignments, 1)
            tIdx = assignments(i, 1);
            dIdx = assignments(i, 2);
            correct(tracks(tIdx).kalmanFilter, centroids(dIdx, :));
            tracks(tIdx).bboxes(end+1, :) = bboxes(dIdx, :);
            tracks(tIdx).age = tracks(tIdx).age + 1;
            tracks(tIdx).totalVisibleCount = ...
                tracks(tIdx).totalVisibleCount + 1;
            tracks(tIdx).scores(end+1) = scores(dIdx);
        end
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).bboxes = ...
                [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores(end+1) = 0;
        end
    end

    function deleteLostTracks()
        if isempty(tracks), return; end
        ages = [tracks(:).age]';
        totalVisible = [tracks(:).totalVisibleCount]';
        visibility = totalVisible ./ ages;
        lostInds = (ages > option.ageThresh & ...
                    visibility < option.visThresh);
        tracks = tracks(~lostInds);
    end

    function createNewTracks()
        for i = 1:length(unassignedDetections)
            idx = unassignedDetections(i);
            kf = configureKalmanFilter( ...
                'ConstantVelocity', centroids(idx,:), ...
                [2, 1], [5, 5], 100);
            newTrack = struct( ...
                'id', nextId, ...
                'color', rand(1,3), ...
                'bboxes', bboxes(idx,:), ...
                'scores', scores(idx), ...
                'kalmanFilter', kf, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'confidence', scores(idx), ...
                'predPosition', bboxes(idx,:) );
            tracks(end + 1) = newTrack;
            nextId = nextId + 1;
        end
    end

    function displayTrackingResults()
        outFrame = frame;
        for i = 1:length(tracks)
            if tracks(i).age > option.ageThresh
                outFrame = insertObjectAnnotation( ...
                    outFrame, 'rectangle', ...
                    tracks(i).bboxes(end,:), ...
                    sprintf('ID:%d', tracks(i).id), ...
                    'Color', tracks(i).color);
            end
        end

        if ~isempty(allIoUs)
            txt = sprintf('IoU medio: %.2f | Dice medio: %.2f', ...
                          mean(allIoUs), mean(allDice));
            outFrame = insertText(outFrame, [10 10], txt, ...
                'FontSize', 16, 'BoxColor', 'yellow', 'TextColor', 'black');
        end

        step(obj.videoPlayer, outFrame);
    end
end
