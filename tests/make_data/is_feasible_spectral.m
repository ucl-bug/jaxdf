function feasible = is_feasible_spectral(x, step_size, order)
% Tests if a combination of inputs, stepsize and derivative order
% is feasible for generating a spectral test
  feasible = true;

  if length(step_size) ~= num_dims(x)
    feasible = false; return;

end
