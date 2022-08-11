% Add kwave to path
addpath(genpath(getenv('KWAVE_PATH')));

% Add folder containing this script to path
[this_path, tmp1, tmp2] = fileparts(mfilename('fullpath'));
addpath(this_path);
clear tmp1 tmp2;

% Check what we are in the same folder as this_path
if ~strcmp(this_path, pwd)
  % Raise error
  error('Please run this script from the same folder as %s', mfilename);
end

% Set random seed to make tests deterministic
rng(42);

% Define possible inputs
input_array = {};
v = zeros(50, 1);
v(25) = 1.0;
input_array{1}  = v;
input_array{2}  = randn(10, 1);
input_array{3}  = randn(11, 1);
input_array{4}  = randn(10, 10);
input_array{5}  = randn(11, 10);
input_array{6}  = randn(11, 11);
input_array{7}  = randn(10, 1) + 1i * randn(10, 1);
input_array{8}  = randn(11, 1) + 1i * randn(11, 1);
input_array{9}  = randn(10, 10) + 1i * randn(10, 10);
input_array{10}  = randn(11, 10) + 1i * randn(11, 10);
input_array{11} = randn(11, 11) + 1i * randn(11, 11);

% Define possible stepsizes
stepsizes = {};
stepsizes{1} = 1.;
stepsizes{2} = 0.1;
stepsizes{3} = [1., 1.];
stepsizes{4} = [0.1, 0.1];
stepsizes{5} = [0.1, 1.0];

% Loop trough all combinaations
id_unique = 0;
for i = 1:length(input_array)
  for j = 1:length(stepsizes)
    id_unique = id_unique + 1;

    % Extract values
    x = input_array{i};
    step_size = stepsizes{j};

    if length(x(:)) == 0
      error('Input is empty');
    end

    % Check if feasible
    if not(is_feasible_spectral(x, step_size, 2))
      continue;
    end
    % Get test structure
    test_data = get_laplacian(x, step_size);

    % Save test_data structure to a json file
    filename = sprintf('../test_data/laplacian_FourierSeries_sz_%d_dims_%d_id_%d.json', ...
      length(x(:)), length(step_size), id_unique);
    disp(['Saving test data to ' filename]);

    % Save test_data structure to a json file
    txt = jsonencode(test_data,PrettyPrint=true);
    fid = fopen(filename, 'w');
    fprintf(fid, txt);
    fclose(fid);
  end
end


%% Local functions
function test_data = get_laplacian(x, step_size)
  % Construct the structure object to be used for testing jaxdf function
  test_data = struct();
  complex_data = ~isreal(x);

  if complex_data
    test_data.x = {real(x), imag(x)};
  else
    test_data.x = {x};
  end
  test_data.step_size = single(step_size);
  test_data.is_complex = complex_data;

  % Get the gradient
  y = laplacian(x, step_size);

  % Save it
  if complex_data
    test_data.y = {real(y), imag(y)};
  else
    test_data.y = {y};
  end
end

function y = laplacian(x, stepsize)
% lifted gradientSpect supporting complex inputs
  if isreal(x)
    if length(stepsize) == 1
      y = gradientSpect(x, stepsize, [], 2);
    elseif length(stepsize) == 2
      [fx, fy] = gradientSpect(x, stepsize, [], 2);
      y = fx + fy;
    elseif length(stepsize) == 3
      [fx, fy, fz] = gradientSpect(x, stepsize, [], 2);
      y = fx + fy + fz;
    end
  else
    if length(stepsize) == 1
      y_r = gradientSpect(real(x), stepsize, [], 2);
      y_i = gradientSpect(imag(x), stepsize, [], 2);
      y = y_r + 1i * y_i;
    elseif length(stepsize) == 2
      [fx_r, fy_r] = gradientSpect(real(x), stepsize, [], 2);
      [fx_i, fy_i] = gradientSpect(imag(x), stepsize, [], 2);
      fx = fx_r + 1i * fx_i;
      fy = fy_r + 1i * fy_i;
      y = fx + fy;
    elseif length(stepsize) == 3
      [fx_r, fy_r, fz_r] = gradientSpect(real(x), stepsize, [], 2);
      [fx_i, fy_i, fz_i] = gradientSpect(imag(x), stepsize, [], 2);
      fx = fx_r + 1i * fx_i;
      fy = fy_r + 1i * fy_i;
      fz = fz_r + 1i * fz_i;
      y = fx + fy + fz;
    end
  end
end
