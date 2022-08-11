function d = num_dims(x)
  % Returns the number of dimensions, correctly identifying 1D arrays as 1D
  if length(size(x)) == 2
    if min(size(x)) == 1
      d = 1;
    else
      d = 2;
    end
  else
    d = length(size(x));
  end
end
