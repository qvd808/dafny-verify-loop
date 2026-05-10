// Spec S: index of a maximum element in a non-empty integer array (BODY filled by generator).

method MaxIndex(a: array<int>) returns (i: int)
  requires a.Length > 0
  ensures 0 <= i < a.Length
  ensures forall k :: 0 <= k < a.Length ==> a[k] <= a[i]
{
<<<BODY>>>
}
