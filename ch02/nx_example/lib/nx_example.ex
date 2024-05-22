defmodule NxExample do
  @doc"""
  mix run -e "NxExample.s2_01()"
  """
  def s2_01 do
    dbg(Nx.tensor([1,2,3]))
  end

  def s2_02 do
    a = Nx.tensor([1,2,3])
    b = Nx.tensor([
      [1,2,3],
      [4,5,6]
    ])
    c = Nx.tensor([
      [
        [
          [
            [1.0, 2]
          ]
        ]
      ]
    ])

    dbg(a)
    dbg(b)
    dbg(c)
  end

  def s2_03 do
    a = Nx.tensor([1,2,3])
    b = Nx.tensor([1.0, 2.0, 3.0])

    dbg(a)
    dbg(b)
  end

  def s2_04 do
    dbg(Nx.tensor(0.0000000000000000000000000000000000000000000001))
  end

  def s2_05 do
    dbg(Nx.tensor(1.0e-45, type: {:f, 64}))
  end

  def s2_06 do
    dbg(Nx.tensor(128, type: {:s, 8}))
  end

  def s2_07 do
    dbg(Nx.tensor([1.0, 2, 3]))
  end

  import Nx.Defn
  defn adds_one(x) do
    Nx.add(x, 1) |> print_expr()
  end

  defn softmax(n) do
    Nx.exp(n) / Nx.sum(Nx.exp(n))
  end

  def benchmark do
    key = Nx.Random.key(42)
    {tensor, _key} =
      Nx.Random.uniform(key, shape: {1_000_000})

    Benchee.run(
      %{
        "JIT with EXLA" => fn ->
          apply(EXLA.jit(&softmax/1), [tensor])
        end,
        "Regular Elixir" => fn ->
          softmax(tensor)
        end
      },
      time: 10
    )

  end
end
