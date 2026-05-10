method MaxArray(a: array<int>) returns (max:int)
requires a.Length > 0
ensures forall i :: 0 <= i < a.Length ==> a[i] <= max
ensures exists i :: 0 <= i < a.Length && a[i] == max
{
<<<BODY>>>
}

method Main() {
	var arr : array<int> := new int[][-11,2,42,-4];
	var res := MaxArray(arr);
	assert arr[0] == -11 && arr[1] == 2 && arr[2] == 42 && arr[3] == -4;
	assert res == 42;
}

