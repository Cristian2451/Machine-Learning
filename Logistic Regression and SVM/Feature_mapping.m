function X_feat = Feature_mapping(x1, x2)
    % Feature mapping function to obtain polynomial feautres. Returns a new
    % array  involving higher order intreactions between the features. The
    % function returns a matrix:
    % [x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, ... ]
    Max_degree = 6;
    X_feat = [];
    for i = 1:Max_degree
        for j = 0:i
            X_feat(:, end+1) = (x1.^(i-j)).*(x2.^j);
        end
    end
end