// Spec S: sorted-array binary search (BODY filled by generator).

predicate Sorted(a: array<int>)
  reads a
{
  forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
}

method BinarySearch(a: array<int>, x: int) returns (index: int)
  requires Sorted(a)
  ensures index >= 0 ==> index < a.Length && a[index] == x
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != x
{
<<<BODY>>>
}
