clear all;
close all;
clc;

G = 200;
NP = 150;
Di = 9;
Pc = 0.8;
Pm = 0.1;
Ps=0.05;
cube=cell(3,3,6);
targetState = cell(3, 3, 6);
fithist = zeros(G, NP);
foundSolution = false;
popu = cell(NP, 1);
fit = zeros(NP, 1);
best_fit_hist=zeros(1,G);
shuff_fit = zeros(Di, 1);
shuff_move=strings(Di,1);
colors = ['W', 'Y', 'O', 'R', 'B', 'G'];
moves = {'F', 'f', 'B', 'b', 'L', 'l', 'R', 'r', 'U', 'u', 'D', 'd'};

for i = 1:6
    color=colors(i);
    cube(:, :, i) = repmat({color}, 3, 3); % 还原状态
    targetState(:,:,i)=repmat({color},3,3);
end

for i=1:Di
    idx=moves{randi(length(moves))};
    fprintf('打乱步骤:%s\n',idx);
    cube=rotate(cube,idx);
    shuff_fit(i)=calcu_entropy(cube,targetState);
    fprintf('步骤 %s: 适应度 = %.4f\n',  idx, shuff_fit(i));
    shuff_move{i}=idx;
end

for i = 1:NP
    popu{i} = char(moves(randi([1, length(moves)], Di, 1))); % 生成Di个随机动作序列
end

cube1 = cube;
max_local_steps = 10; % 局部搜索最大步数
local_search_rate = 0.2; % 进行局部搜索的概率
best_solution = [];
best_fitness = inf;

for gen = 1:G
    for i = 1:NP
        temp_cube = cube1; % 每个个体从当前状态开始转动
        for step = popu{i}' % 按个体的转动序列转动
            temp_cube = rotate(temp_cube, step);
        end
        fit(i) = calcu_entropy(temp_cube,targetState);
    end
    fithist(gen, :) = fit;
    best_fit_hist(gen) = min(fit);
    [min_fit, min_idx] = min(fit);
    fprintf('第 %d 代最优适应度: %.4f,个体(%d):%s\n', gen, best_fit_hist(gen),min_idx,num2str(popu{min_idx}));

    if best_fit_hist(gen) == 0
        disp('魔方已复原！');
        foundSolution = true;
        break;
    end
    [popu, fit] = elitism(popu, fit);
    % elite_count = 150;
    for i = 1:20
        if rand < local_search_rate
            [popu{i}, fit(i)] = local_search(popu{i}, fit(i), cube1, targetState, max_local_steps, moves);
        end
    end
    [popu, fit] = cro(popu, cube1, Pc, NP, shuff_fit, targetState, fit,Di); % 交叉
    [popu, fit] = mutate(popu, cube1, fit, Pm, Ps, Di, moves, targetState, NP,shuff_fit); % 变异

end
function [best_sequence, best_fit] = local_search(sequence, current_fit, cube1, targetState, max_steps, moves)
best_sequence = sequence;
best_fit = current_fit;
for step1 = 1:max_steps
    new_sequence = sequence;
    operation = randi(3);
    if operation == 1 % 翻转
        idx1 = randi(length(new_sequence));
        idx2 = randi(length(new_sequence));
        start_idx = min(idx1, idx2);
        end_idx = max(idx1, idx2);
        new_sequence(start_idx:end_idx) = flip(new_sequence(start_idx:end_idx));
    elseif operation == 2 % 替换
        replace_idx = randi(length(new_sequence));
        new_move = moves{randi(length(moves))};
        new_sequence(replace_idx) = new_move;
    elseif operation == 3 % 交换
        idx1 = randi(length(new_sequence));
        idx2 = randi(length(new_sequence));
        temp = new_sequence(idx1);
        new_sequence(idx1) = new_sequence(idx2);
        new_sequence(idx2) = temp;
    end
    cube_state = cube1;
    for step = new_sequence'
        cube_state = rotate(cube_state, step);
    end
    new_fit = calcu_entropy(cube_state, targetState);
    if new_fit < best_fit
        best_fit = new_fit;
        best_sequence = new_sequence;
    end
    if best_fit == 0
        break;
    end
end
end

for y=1:Di
    fprintf('打乱步骤：%s\n',shuff_move{y});
end

function [popu, fit] = elitism(popu, fit)
[~, sorted_idx] = sort(fit); % 按适应度升序排序，获取索引
elite_idx = sorted_idx(1:15); % 精英个体索引
non_elite_idx = setdiff(1:length(popu), elite_idx);
shuffled_idx = non_elite_idx(randperm(length(non_elite_idx)));
elite_popu = popu(elite_idx, :); % 精英个体
elite_fit = fit(elite_idx); % 精英个体适应度
shuffled_popu = popu(shuffled_idx, :); % 打乱后的非精英个体
shuffled_fit = fit(shuffled_idx); % 对应的适应度
popu = [elite_popu; shuffled_popu];
fit = [elite_fit; shuffled_fit];
end


function H = calcu_entropy(currentState,targetState)
correctColorCount = 0;
correctPositionCount = 0;
totalColorCount = 6 * 3 * 3;  % 每个面9个颜色，总共6个面
for face = 1:6
    faceState = currentState(:, :, face);
    targetFace = targetState(:, :, face);
    correctColorCount = correctColorCount + sum(strcmp(faceState(:), targetFace(:)));
    for row = 1:3
        for col = 1:3
            if strcmp(faceState(row, col) , targetFace(row, col))
                correctPositionCount = correctPositionCount + 1;
            end
        end
    end
end
H = 1 - ((0.6 * correctPositionCount + 0.4 * correctColorCount) / totalColorCount);
end

function [popu, fit] = cro(popu, cube1, Pc, NP, shuff_fit, targetState, fit,Di)
offspring = cell(NP, 1);  % 初始化子代
cro_fit = zeros(NP, 1);
cro_cube = cell(NP, 1);
for i = 1:2:NP
    r3 = rand();
    parent1 = popu{i};
    parent2 = popu{i + 1};
    fit_P1 = zeros(length(parent1), 1);
    fit_P2 = zeros(length(parent2),1);
    cube_state1 = cube1;
    cube_state2 = cube1;
    for step_idx1 = 1:length(parent1)
        cube_state1 = rotate(cube_state1, parent1(step_idx1));
        fit_P1(step_idx1) = calcu_entropy(cube_state1, targetState);
    end
    for step_idx2 = 1:length(parent2)
        cube_state2 = rotate(cube_state2, parent2(step_idx2));
        fit_P2(step_idx2) = calcu_entropy(cube_state2, targetState);
    end
    if r3 < Pc  % 如果进行交叉
        child1 = parent1;
        child2 = parent2;
        m1 = 0;
        m2 = 0;
        C1=0;
        C2=0;
        for j = 1:length(parent1) - 1
            if fit_P1(j) ~= shuff_fit(Di-1)
                m1 = j; % 记录不匹配点
                break;
            else
                C1=C1+1; % 若匹配，更新为当前点
            end

            if fit_P2(j) ~= shuff_fit(Di-1)
                m2 = j; % 记录不匹配点
                break;
            else
                C2=C2+1; % 若匹配，更新为当前点
            end
        end
        m = max(m1, m2);
        if C1==0 && C2==0
            if Di <= 3
                num_cro_point = 1;  % 交叉点数量
                cross_points=sort(randperm(Di,num_cro_point));
                temp = child1(cross_points);
                child1(cross_points) = child2(cross_points);
                child2(cross_points) = temp;
            else
                num_cro_point = randi([2, floor(Di/2)]);  % 交叉点数量
                cross_points=sort(randperm(Di-1,num_cro_point));
                for j = 1:length(cross_points) - 1
                    start_idx = cross_points(j) ;
                    end_idx = cross_points(j+1)-1;
                    temp = child1(start_idx:end_idx);
                    child1(start_idx:end_idx) = child2(start_idx:end_idx);
                    child2(start_idx:end_idx) = temp;
                end
            end
        else
            cross_start = m;
            max_cro_point = length(parent1) - cross_start;
            num_cro_point = randi([2, max_cro_point]);
            cross_points = sort(randperm(length(parent1) - cross_start, num_cro_point) + cross_start);
            for j = 1:length(cross_points) - 1
                start_idx = cross_points(j);
                end_idx = cross_points(j+1)-1;
                temp = child1(start_idx:end_idx);
                child1(start_idx:end_idx) = child2(start_idx:end_idx);
                child2(start_idx:end_idx) = temp;
            end
        end
        offspring{i} = child1;
        offspring{i + 1} = child2;
    else
        offspring{i} = parent1;
        offspring{i + 1} = parent2;
    end
end
for j = 1:NP
    cube3 = cube1;
    for step = offspring{j}'
        cube3 = rotate(cube3, step);
    end
    cro_cube{j} = cube3;
    cro_fit(j) = calcu_entropy(cro_cube{j}, targetState);
    if cro_fit(j) < fit(j)
        popu{j} = offspring{j};
        fit(j) = cro_fit(j);
    end
end
end

function [popu, fit] = mutate(popu, cube1, fit, Pm, Ps, Di, moves, targetState, NP,shuff_fit)
mutated = cell(NP, 1);
mut_fit = zeros(NP, 1);
mut_cube = cell(NP, 1);
P=0;
for i = 1:NP
    sequence = popu{i};
    fit_M1 = zeros(length(sequence), 1);
    cube_state3 = cube1;
    r3=rand();
    if r3 < Pm
        for step = 1:Di
            cube_state3 = rotate(cube_state3, step);
            fit_M1(step) = calcu_entropy(cube_state3, targetState);
        end
        mutated_flag = false;
        r4=rand();
        if r4<Ps
            for j=1:length(sequence)-1
                if fit_M1(j) ~= shuff_fit(Di-1)
                    original_move = sequence(j);
                    new_move = original_move;
                    while isequal(new_move, original_move)
                        new_move = moves(randi(length(moves)));
                    end
                    sequence(j) = char(new_move); % 更新当前步骤的变异操作
                    mutated_flag = true;
                else
                    P=P+1;
                end
                if P == length(sequence) - 1
                    original_move = sequence(end);
                    new_move = original_move;
                    while isequal(new_move, original_move)
                        new_move = moves(randi(length(moves))); % 随机选择一个新的动作
                    end
                    sequence(end) = char(new_move); % 更新最后一步的变异操作
                    mutated_flag = true; % 标志发生了变异
                end
                if ~mutated_flag
                    original_move = sequence(end); % 最后一操作
                    new_move = original_move;
                    while isequal(new_move, original_move) % 确保新操作不同
                        new_move = moves(randi(length(moves))); % 随机生成新的转动操作
                    end
                    sequence(end) = char(new_move); % 更新最后一步操作
                end

            end
        end
        mutated{i} = sequence;
    else
        mutated{i} = sequence; % 保存变异后的序列
    end
end
for j = 1:NP
    cube4 = cube1;
    for step = mutated{j}' % 按变异序列依次旋转
        cube4 = rotate(cube4, step);
    end
    mut_cube{j} = cube4;
    mut_fit(j) = calcu_entropy(mut_cube{j},targetState);
    if mut_fit(j) < fit(j)
        popu{j} = mutated{j};
        fit(j) = mut_fit(j);
    end
end
end

function currentState=rotate(currentState,move)
rotate_temp=currentState;
switch move
    % 1-前-白,2-后-黄,3-左-橙,4-右-红,5-上-蓝,6-下-绿
    case 'F'
        temp = currentState(:,3,3);
        rotate_temp(:,3,3) = rotate_temp(1,:,6);
        rotate_temp(1,:,6) = flipud(rotate_temp(:,1,4));
        rotate_temp(:,1,4) = rotate_temp(3,:,5);
        rotate_temp(3,:,5) = flipud(temp);
        rotate_temp(:,:,1) = manualRotate(rotate_temp(:,:,1), 1);
    case 'f'
        temp = rotate_temp(:,3,3);
        rotate_temp(:,3,3) = flipud(rotate_temp(3,:,5)');
        rotate_temp(3,:,5) = rotate_temp(:,1,4);
        rotate_temp(:,1,4) = flipud(rotate_temp(1,:,6)');
        rotate_temp(1,:,6) = temp;
        rotate_temp(:,:,1) = manualRotate(rotate_temp(:,:,1), -1);
    case 'B'
        temp = rotate_temp(:,1,3);
        rotate_temp(:,1,3) = rotate_temp(3,:,6);
        rotate_temp(3,:,6) = flipud(rotate_temp(:,3,4));
        rotate_temp(:,3,4) = rotate_temp(1,:,5);
        rotate_temp(1,:,5) = flipud(temp);
        rotate_temp(:,:,2) = manualRotate(rotate_temp(:,:,2), -1);
    case 'b'
        temp = rotate_temp(:,1,3);
        rotate_temp(:,1,3)= flipud(rotate_temp(1,:,5)');
        rotate_temp(1,:,5)= flipud(rotate_temp(:,3,4)');
        rotate_temp(:,3,4)= flipud(rotate_temp(3,:,6)');
        rotate_temp(3,:,6)= flipud(temp');
        rotate_temp(:,:,2) = manualRotate(rotate_temp(:,:,2), 1);
    case 'L'
        temp = rotate_temp(:,1,1);
        rotate_temp(:,1,1) = rotate_temp(:,1,5);
        rotate_temp(:,1,5) = flipud(rotate_temp(:,3,2));
        rotate_temp(:,3,2) = flipud(rotate_temp(:,1,6));
        rotate_temp(:,1,6) = temp;
        rotate_temp(:,:,3) = manualRotate(rotate_temp(:,:,3), 1);
    case 'l'
        temp = rotate_temp(:,1,1);
        rotate_temp(:,1,1) = rotate_temp(:,1,6);
        rotate_temp(:,1,6) = flipud(rotate_temp(:,3,2));
        rotate_temp(:,3,2) = flipud(rotate_temp(:,1,5));
        rotate_temp(:,1,5) = temp;
        rotate_temp(:,:,3) = manualRotate(rotate_temp(:,:,3), -1);
    case 'R'
        temp = rotate_temp(:,3,1);
        rotate_temp(:,3,1) = rotate_temp(:,3,6);
        rotate_temp(:,3,6) = flipud(rotate_temp(:, 1, 2));
        rotate_temp(:,1,2) = flipud(rotate_temp(:,3,5));
        rotate_temp(:,3,5) = temp;
        rotate_temp(:,:,4) = manualRotate(rotate_temp(:,:,4), 1);
    case 'r'
        temp = rotate_temp(:,3,1);
        rotate_temp(:,3,1) = rotate_temp(:,3,5);
        rotate_temp(:,3,5) = flipud(rotate_temp(:, 1, 2));
        rotate_temp(:,1,2) = flipud(rotate_temp(:,3,6));
        rotate_temp(:,3,6) = temp;
        rotate_temp(:,:,4) = manualRotate(rotate_temp(:,:,4), -1);
    case 'U'
        temp = rotate_temp(1,:,1);
        rotate_temp(1,:,1) = rotate_temp(1,:,4);
        rotate_temp(1,:,4) = flipud(rotate_temp(1,:,2));
        rotate_temp(1,:,2) = flipud(rotate_temp(1,:,3));
        rotate_temp(1,:,3) = temp;
        rotate_temp(:,:,5) = manualRotate(rotate_temp(:,:,5), 1);
    case 'u'
        temp = rotate_temp(1,:,1);
        rotate_temp(1,:,1) = rotate_temp(1,:,3);
        rotate_temp(1,:,3) = flipud(rotate_temp(1,:,2));
        rotate_temp(1,:,2) = flipud(rotate_temp(1,:,4));
        rotate_temp(1,:,4) = temp;
        rotate_temp(:,:,5) = manualRotate(rotate_temp(:,:,5), -1);

    case 'D'
        temp = rotate_temp(3,:,1);
        rotate_temp(3,:,1) = rotate_temp(3,:,4);
        rotate_temp(3,:,4) = flipud(rotate_temp(3,:,2));
        rotate_temp(3,:,2) = flipud(rotate_temp(3,:,3));
        rotate_temp(3,:,3) = temp;
        rotate_temp(:,:,6) = manualRotate(rotate_temp(:,:,6), -1);
    case 'd'
        temp = rotate_temp(3,:,1);
        rotate_temp(3,:,1) = rotate_temp(3,:,3);
        rotate_temp(3,:,3) = flipud(rotate_temp(3,:,2));
        rotate_temp(3,:,2) = flipud(rotate_temp(3,:,4));
        rotate_temp(3,:,4) = temp;
        rotate_temp(:,:,6) = manualRotate(rotate_temp(:,:,6), 1);
end
currentState=rotate_temp;
end

% 更新转动面颜色
function rotatedFace = manualRotate(face, direction)
if direction == 1  % 顺时针旋转
    rotatedFace = fliplr(face');% 矩阵转置后翻转列
elseif direction == -1  % 逆时针旋转
    rotatedFace = flipud(face'); % 矩阵转置后翻转行
else
    rotatedFace = face; % 未旋转
end
end

figure;
plot(1:gen, best_fit_hist(1:gen), 'LineWidth', 1);
xlabel('迭代次数');
ylabel('最优适应度');
title('最优适应度变化曲线');
grid on;
