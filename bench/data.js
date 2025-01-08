window.BENCHMARK_DATA = {
  "lastUpdate": 1736339114793,
  "repoUrl": "https://github.com/JuliaGPU/Metal.jl",
  "entries": {
    "Metal Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "distinct": true,
          "id": "71b784ec9d6adf8a758e4d77f3b12ed53e7cea34",
          "message": "Temporarily allow benchmark runs on all workers.\n\n[only benchmarks]",
          "timestamp": "2024-10-02T09:50:18+02:00",
          "tree_id": "1e15b5261c536d7ff8d5ef15b90409f342ae0db6",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/71b784ec9d6adf8a758e4d77f3b12ed53e7cea34"
        },
        "date": 1727856593123,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 23715.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 474145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 994125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 644458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 569958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 606250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 831750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 897625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 660666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 555208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 709417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1430125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1499500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2210520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2041209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1704833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1645334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2430625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3432895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1763667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1353479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 730853.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 709708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 800041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 713125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 949333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 930958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1018708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 582583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4403995333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6895957979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 723655188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 757604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1623541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8853854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1573521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2624459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 455583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 461916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 19791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 23972.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 478708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4912\nallocs=179\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 987500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 641062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 576520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 592333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 870458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 935229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 546667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 94125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1434979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1497729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1971125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1777500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1410291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1388708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 189562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3205291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1479229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1373083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26632\nallocs=869\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 616666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 716854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 686417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 710584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 960250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 925458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1015208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 598354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tgymnich@icloud.com",
            "name": "Tim Gymnich",
            "username": "tgymnich"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ff7c7ebdcd5f6d513614e3a27062d06a07ce81f7",
          "message": "Use unified memory for scalar indexing of permutation matrices (#313)\n\nCo-authored-by: Tim Besard <tim.besard@gmail.com>",
          "timestamp": "2024-10-02T14:14:21+02:00",
          "tree_id": "979007b2651f0a3bbacd0edd2379af4efb2cc7f5",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/ff7c7ebdcd5f6d513614e3a27062d06a07ce81f7"
        },
        "date": 1727874377921,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 465979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 993270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 632166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 568500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 583500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 880458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 844333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 614333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 739479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 599208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1447750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1496375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2263917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1989875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1678000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1663625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2393834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3431520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1794125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1403416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 805792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 704146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 815812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 716666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 943959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 938875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1005416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 862875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4407793041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6915521687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 726643917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 749270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1557959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8832020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1611291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2583750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 476584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 441500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 19750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 23756.916666666664,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 469584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 1020166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 634458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 572000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 593208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 742792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 898812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 659667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 94458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1418250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1500167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1939666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1746333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1413458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1374750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 189167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3212770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1481709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1379250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 659583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 706354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 620667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 704958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 963438,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 939020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1003520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 880541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "da83511095b183728e47e2adc14079f5e3c120e6",
          "message": "Disable nightly CI and fix invalid Metal API usage (#448)",
          "timestamp": "2024-10-04T10:13:55+02:00",
          "tree_id": "a220ced490cbb715e87750be20d3e728bf135421",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/da83511095b183728e47e2adc14079f5e3c120e6"
        },
        "date": 1728035589087,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26465.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 461042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 1011750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 641708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 573459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 602667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 861583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 878333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 473041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 713958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 442208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1437271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1483417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2255250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2046124.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50360\nallocs=1846\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1680458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1669021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2356917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3450375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1770125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1371417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 750333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 723583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 791625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 713708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 957895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 944583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 994792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 861917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4407307208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6930583500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 723401145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 746250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1556458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8821042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1563958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2552084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 470917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 446250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 11791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 19833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 23930.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 453646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 1017833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 649500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 572417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 613583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 813417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 847083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 658542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 84500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 85916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1449250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1494791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2003708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1778291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1418958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1386270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 193750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3214083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1474146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1391229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 680375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 711146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 673729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 714021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 958167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 947375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 999917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 856125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cd9e68a1ffd37e8cf8c86af81acefed56f371f0b",
          "message": "Don't report benchmarks on main branch commits (#450)\n\n[only benchmarks]",
          "timestamp": "2024-10-04T15:05:20-03:00",
          "tree_id": "92ee9bd1278fccd97052c5bb03ab0061d2bf9850",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/cd9e68a1ffd37e8cf8c86af81acefed56f371f0b"
        },
        "date": 1728074161106,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26732.583333333336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 461834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 1007750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 633312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 581542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 592916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 870916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 977958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 486875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 720666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 576000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1445354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1477250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2238875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2031625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1678291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1671396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2361083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3433396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1770396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1363729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 724645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 711916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 798667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 714500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 946542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 937833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1007167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 893375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4410766833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6887075395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 725464062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 755125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1577625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8895417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1559209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2635958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 462375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 446500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 11833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 19833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 24125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 467083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 1016896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 634250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 573167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 592958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 865103.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 831271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 678792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 94625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1446583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1483229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1952167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1778083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1415604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1376542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 192125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3214833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1477146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1384958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 644417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 700500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 673667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 710479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 935500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 944334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1023583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 808959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46298cad934c43caa525a092d2668bf51ec629e1",
          "message": "Fix unsafe_wrap of a view. (#452)\n\nAnd some other small fixes.",
          "timestamp": "2024-10-05T13:55:15+02:00",
          "tree_id": "314a2bcc453ee62e76e425f9585df9a65f54cfc8",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/46298cad934c43caa525a092d2668bf51ec629e1"
        },
        "date": 1728132347494,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 25739.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 458958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 1002854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 636250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 573500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 607375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 869584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 867417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 488708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 676271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 548688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1439583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1476083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2256417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2053916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1656375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1643709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2401417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3425750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1773625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1368042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 799187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 704583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 842458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 708020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 946458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 944750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 990416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 850750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4400518084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6886083333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 723478145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 753417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1585834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8919167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1547541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2641479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 467500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 444417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 11125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 19750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 24263.833333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 468875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 1017250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 648500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 574375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 606417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 843500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 866145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 596334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 84000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 83083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1450166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1487041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1963687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1783187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1414208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1386312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 189250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3167645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1477541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1386291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 685375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 707167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 677292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 707000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 955687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 940958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1018396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 768750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c4c0e28302380dd63aff9cb308c099157510c50d",
          "message": "Use CPU memcpy for SharedStorage copyto! (#445)",
          "timestamp": "2024-10-08T10:27:03+02:00",
          "tree_id": "51555f01cddbaa3f7a94105a2e8dbf8f72317172",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/c4c0e28302380dd63aff9cb308c099157510c50d"
        },
        "date": 1728378303530,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26854.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 755020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 615958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 546708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 580395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 775979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 613959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 634458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 763959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 586667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1322145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1413437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2100833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1836853.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1763166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1640479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3154645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3261458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1545667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1328896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1068666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 700396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1047125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 692042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 841000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 843750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 917750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 647874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4401597375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6711178812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 720307000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 743250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1607375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 11412000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1562958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2659937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11552\nallocs=455\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 470417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 478812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 23483.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":5,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 465250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 780542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 645041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 549166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 581208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 791375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 627646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 89250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1120\nallocs=54\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 85000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 80792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1351083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1404792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1809750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1578708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1384250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1359834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 149792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3048854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1279854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1341854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 680542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 698666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 725209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 699125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 847916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 866729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 941229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 247125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2744\nallocs=91\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dbed7500692b8587cbcc78048386048e0e04f365",
          "message": "Only load BFloat16sExt on Apple systems (#454)",
          "timestamp": "2024-10-10T16:13:10-03:00",
          "tree_id": "a4aa490413fea3afc9c0dbe93c7e5cd252df94d4",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/dbed7500692b8587cbcc78048386048e0e04f365"
        },
        "date": 1728590651520,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26291.666666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 467021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 753375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 671520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 583625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 605396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 822187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 573083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 496250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 721500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 547709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1449000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1485208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2232249.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2050021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1710500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1672020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2420750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3460208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1811500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1375271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 847500.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 715458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 859542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 723104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 951667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 935062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1007666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 510583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4390918458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6919873812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 720849750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 762834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1655833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8906833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1551959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2757562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 470270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 451542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 11583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 20209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 24708.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 463042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 792208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 664166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 586062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 610958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 831208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 592125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 93125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1120\nallocs=54\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 92917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1443292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1480896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2014375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1752250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=49744\nallocs=1822\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1410000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1376812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 196709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3222917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1484375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1391000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 697000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 709667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 704312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 767125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 939312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 915250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1002479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 283250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2744\nallocs=91\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41898282+github-actions[bot]@users.noreply.github.com",
            "name": "github-actions[bot]",
            "username": "github-actions[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddca5c4f08f24187885556fc498856c6791bceda",
          "message": "CompatHelper: bump compat for GPUCompiler to 1, (keep existing compat) (#455)\n\nCo-authored-by: CompatHelper Julia <compathelper_noreply@julialang.org>",
          "timestamp": "2024-10-11T07:50:08+02:00",
          "tree_id": "f8f9b21f1991243f90e3dce441c035eec3c5b44c",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/ddca5c4f08f24187885556fc498856c6791bceda"
        },
        "date": 1728628035492,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26722.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 467500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 751250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 636125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 578875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 589417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 813250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 542500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 515812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 727666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 554625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1444334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1469333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2243208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2115125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1690562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1674250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2412750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3443541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=84632\nallocs=3107\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1771104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1362584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 793250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 713896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 864750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 712500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 912916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 931604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1000937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 509083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4397344834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6908319354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 719607250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 756958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1632792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8919542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1659354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2684500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 478708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 476750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 12084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 19916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 20209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 24899.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 470042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 754416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 650708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 586083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 595750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 809708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 589562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 93459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1120\nallocs=54\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 95291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1440417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1482250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1940666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1770833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1395458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1371084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 195250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3219395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1497458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1371541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 638333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 718750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 708500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 707208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 952146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 929292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1039791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 288396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2744\nallocs=91\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "100f8318a60fcbbbde1964c304f95d9419ee1928",
          "message": "Fix loading on unsupported platforms (#459)\n\n* Gate getpagesize call behind function\r\n\r\n* Make `functional` require a fully-supported GPU",
          "timestamp": "2024-10-16T15:16:50-03:00",
          "tree_id": "b64711bdc147fb5573ee1d0cebb822c26ccfb9a3",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/100f8318a60fcbbbde1964c304f95d9419ee1928"
        },
        "date": 1729105215646,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26340.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 465791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 827937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 635250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 562291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 594500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 801791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 591208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 645500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1200\nallocs=58\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 618187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 798500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=73\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1333083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41936\nallocs=1544\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1424500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11800\nallocs=435\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2100167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=57048\nallocs=2078\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1847000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=50056\nallocs=1845\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1696166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1651958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23952\nallocs=851\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3657771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17056\nallocs=701\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3264437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83248\nallocs=3066\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1565166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22032\nallocs=776\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1351333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1063291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 695645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1078084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18336\nallocs=689\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 705166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 860084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 862229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 919520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 574854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2824\nallocs=95\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4396587542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6698494124.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 722852834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 719875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1530167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7472\nallocs=289\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9115541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15824\nallocs=620\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1520271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6984\nallocs=278\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2666416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11392\nallocs=454\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 468541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4664\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 461292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4672\nallocs=185\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=48\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=224\nallocs=9\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=368\nallocs=17\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26069.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 468333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4752\nallocs=178\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 785583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 626541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 564084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 598792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=768\nallocs=34\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 788666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 629791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 96916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1120\nallocs=54\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 88583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 83458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=720\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1356667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=41920\nallocs=1543\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1421333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11784\nallocs=434\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1792833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=55688\nallocs=2024\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1620166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=48696\nallocs=1791\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1385791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1376291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22576\nallocs=798\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 151458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3616\nallocs=171\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3042333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=83272\nallocs=3053\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1274875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20656\nallocs=723\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1346333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26472\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 694458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 702292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 754229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16992\nallocs=636\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 705395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8808\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 858875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9920\nallocs=350\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 862292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9520\nallocs=346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 923916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9720\nallocs=348\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 246583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2744\nallocs=91\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c953739eb6fda54b5c6239fe7b4cae96dea3f9e8",
          "message": "Remove old code and clean-up tests (#464)",
          "timestamp": "2024-10-19T08:46:54+02:00",
          "tree_id": "f0358c18d581a0d2e4466ba7a200c7700777d500",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/c953739eb6fda54b5c6239fe7b4cae96dea3f9e8"
        },
        "date": 1729323717201,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27302.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 464542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4880\nallocs=186\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 828520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 665000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 583958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 622708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 800646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 622292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 719708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1392\nallocs=69\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 532375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2448\nallocs=83\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 761083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2448\nallocs=83\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1385770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45000\nallocs=1661\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1421333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11928\nallocs=443\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2173833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=60288\nallocs=2206\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1899500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=53216\nallocs=1968\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1674542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=24224\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1641375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=24224\nallocs=868\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3935416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18176\nallocs=771\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3285624.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=89136\nallocs=3291\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1565229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22352\nallocs=796\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1333667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26760\nallocs=886\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1067374.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18496\nallocs=707\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 678125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8776\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1090458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18496\nallocs=707\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 675959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8776\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2919166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13368\nallocs=415\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1048417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11560\nallocs=405\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1612625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=407\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 620333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3016\nallocs=106\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 4510050000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6891418604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 879412875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1312\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 718604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7112\nallocs=286\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1570417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7600\nallocs=297\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9622874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15952\nallocs=628\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1560084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7112\nallocs=286\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2487792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11520\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 491104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4792\nallocs=193\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 472750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4800\nallocs=193\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1472\nallocs=53\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=12\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=416\nallocs=20\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25784.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1264\nallocs=35\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 456500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4880\nallocs=186\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 797209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 691500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 580375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 616291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=912\nallocs=43\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 739770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 636312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2432\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 88708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1216\nallocs=60\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 86000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 79334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=38\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1397709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44984\nallocs=1660\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1416959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11912\nallocs=442\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1896416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=59912\nallocs=2179\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1653228.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=51792\nallocs=1910\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1389478.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22784\nallocs=811\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1371062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22784\nallocs=811\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 160750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4096\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3047833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=87712\nallocs=3233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1278291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=20912\nallocs=739\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1342625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=26760\nallocs=886\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 711500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17088\nallocs=650\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 677791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8776\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 758667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17088\nallocs=650\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 677250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8776\nallocs=319\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2921500.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13368\nallocs=415\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1044792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11560\nallocs=405\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1603709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=407\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 251979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2840\nallocs=97\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "15ac66d722f1e83584a127bde61b47e4d91d83f8",
          "message": "Switch CI to 1.11. (#462)",
          "timestamp": "2024-10-19T20:02:23+02:00",
          "tree_id": "0ad8178685a124168abd87706d12a0116b6babfd",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/15ac66d722f1e83584a127bde61b47e4d91d83f8"
        },
        "date": 1729363490141,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27827.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 465541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 831500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 656625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 548229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 596000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 772125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 616292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 705667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 630625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 825292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1304812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1382792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2027166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45944\nallocs=1652\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1816875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40624\nallocs=1490\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1688271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1629458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3825958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3194354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1736437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1346833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1045562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 647334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1051375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 656291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2701896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1005771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1561083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 599500.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5210869875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6541068271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1140328708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 694000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1586291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10928917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1622646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2588083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 488771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 478083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25173.583333333336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 470833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 785625.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 661417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 576917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 612500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 809208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 625083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 88083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1339709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1382229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1775500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44664\nallocs=1609\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1593041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1388583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1350167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 157125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2964584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1459458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1367083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 740209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 664125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 748291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 662708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2702396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1011791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1577625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 245125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9019b56c05055db3e5dcd93ca0d08bf264c908cd",
          "message": "Adapt to JuliaGPU/GPUArrays.jl#567. (#475)",
          "timestamp": "2024-10-30T14:28:49+01:00",
          "tree_id": "e49a1b0d9881cf501bf4c7b3d9dfa3f540a62e0f",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/9019b56c05055db3e5dcd93ca0d08bf264c908cd"
        },
        "date": 1730297444386,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26791.714285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 456167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 822916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 623604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 546542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 587084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 751458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 567021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 486625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 810666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 614333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1326250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1382709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2047625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1824979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1694750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1644604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3778833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3169979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1756542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1335208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1020375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 650833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1036291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 652916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 3071292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 998625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9184\nallocs=311\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1578500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 586396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5193067083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6487019417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1129040229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 699833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1611416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 11381521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1574875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2675708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 453209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 460187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 36083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 462875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 806354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 647333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 549333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 583750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 769417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 639209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 84917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 82208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1353834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1379437,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1792708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1589958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1398958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1356875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 155500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2962771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1464208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1356667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 722708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 653541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 740083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 655334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2936646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1005417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9184\nallocs=311\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1572312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 243917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "240c4b0988abf6cd6337072141ceb55ad9699ad2",
          "message": "Bump LLVM downgrader (#479)",
          "timestamp": "2024-11-08T09:20:40+01:00",
          "tree_id": "ea7a6a57627b091e30259ec6c541b4f65a60a06c",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/240c4b0988abf6cd6337072141ceb55ad9699ad2"
        },
        "date": 1731056647568,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 28101.285714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 459125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 793875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 559916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 585959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 781459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 594542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 633792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 792521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 674167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1332000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1375083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2087917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45688\nallocs=1653\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1820166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=40624\nallocs=1490\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1693770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1657375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3841520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3160687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1755333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1343458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1029271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 655542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1052084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 653937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 3078396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 989042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1572500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 617250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5189463875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6484573312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1127755937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 727750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1603458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10250541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1591771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5208\nallocs=206\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2610104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8608\nallocs=343\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 452792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 450875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 9250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26208.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 454562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 820333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 678895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 559834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 595854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 757584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 624229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 86834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1347875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1385042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1772479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45240\nallocs=1630\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1569917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1406542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1364375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 150542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2973291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1460375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1359417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 724500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 659500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 747542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 656792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 3090875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1004542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1574188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 233916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3ec440f5540d372f02660cc7c59f29458ab0bf3e",
          "message": "Store more debug files when encountering compilation errors. (#482)",
          "timestamp": "2024-11-27T22:57:49+01:00",
          "tree_id": "7b30f0f91da1532e35240e79077bda789051572f",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/3ec440f5540d372f02660cc7c59f29458ab0bf3e"
        },
        "date": 1732747247292,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 25494,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 464917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 913875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 595833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 551583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 558666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 899583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 817333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 561250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 702791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 621542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1408542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1502750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2276334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1993104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39864\nallocs=1465\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1808458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1712312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2520292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3486084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1882417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1417750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 964500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 697917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 997020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 693750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 3093667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1068458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1842166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 836458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5253931625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6649813771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1145857916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 750770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1662375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 20314209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1666875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2757583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 460145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 463042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 35656.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25805.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 461791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 919417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 600250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 548958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 561916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 876166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 824312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 80334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 81084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 79125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1417750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1499520.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2022521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1713854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1489417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1415500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 159208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3167979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1558916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1427437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21656\nallocs=717\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 697458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 697125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 707292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 702291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 3157708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1067771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1874583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 206458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8ccd416c47d335db3934d3c22f1d28e06ba6ff30",
          "message": "Still run GH Action when merged (#486)",
          "timestamp": "2024-11-30T20:42:33-04:00",
          "tree_id": "1624ce7df283973ecc907fafd491f84500b60b95",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/8ccd416c47d335db3934d3c22f1d28e06ba6ff30"
        },
        "date": 1733016273103,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27007,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 856292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 637833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 561708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 595541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 786833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 647312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 643188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 670333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 825375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1324916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1391416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2065417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1825542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1718500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1648583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3915916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3168416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1741000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1354250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1046625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 653625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1057708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 649708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2704229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 996500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1568125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 614500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5186888417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6501679688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1131305770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 695667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1628959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10528167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1564708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2716709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 455416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 450645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 469000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 767396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 669417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 563208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 598333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 764250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 634250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 88792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 89042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 83083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1366917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1384500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1809083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45240\nallocs=1630\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1587541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1391729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1357708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 160959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2952208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1457458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1367167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 733792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 656500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 746958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 654208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2692625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1004125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1570166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 239625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3fd6c06456dfdea8fa73483d396178a3202db666",
          "message": "Use `OncePerProcess` when possible (#483)",
          "timestamp": "2024-12-02T15:25:33+01:00",
          "tree_id": "aafd39d90f0ef12538d08ae2847c6ae5281173f1",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/3fd6c06456dfdea8fa73483d396178a3202db666"
        },
        "date": 1733152290657,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27095.285714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 465000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 789812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 660666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 572250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 597625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 736833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 631958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 653750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 817417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 634167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1339791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1393417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2094937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1826875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1686499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1657166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3913229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3178375.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1756417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1353625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1039812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 661333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1046500.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 658292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2708916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1008083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1583375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 579458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 6399118167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6579444208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1130024792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 710542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1546084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9416354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1601166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2699792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 457917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 449833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10034.833333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25840.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 457229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 796187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 647958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 563875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 601416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 748833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 603916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 86542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 88375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 78375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1350249.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1390125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1793917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1592959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1401625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1363333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 156875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2962417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1463291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1365959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 730375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 661375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 740833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 662354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2838125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1005479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1562084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 247562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e6abcc63d2924914d79803e52ac0fe31d52cacdd",
          "message": "Bump IR downgrader (#489)",
          "timestamp": "2024-12-05T08:42:11+01:00",
          "tree_id": "4c748331379e2fc8dec3c62063d316fc8ec41930",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/e6abcc63d2924914d79803e52ac0fe31d52cacdd"
        },
        "date": 1733395199921,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26616.071428571428,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 460792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 769959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 646500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 546000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 586458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 777000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 585875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 659812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 823645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 624479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1332291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1376666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2040417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1822812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1681646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1650958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3843750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3170916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1762708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1330854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1018354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 657208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1025958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 650333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2706375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 989125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1562396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 599500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5119209208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6736074937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1150978708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 712813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1640250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10578833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1533458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2647313,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 493833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 484521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 9889,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26827.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 453250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 762499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 644042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 538500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 570792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 769625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 645625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 89416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 86458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1335958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1384187,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1755958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1576938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1393500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1352667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 154625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2967917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64152\nallocs=2363\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1454875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1360000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 715000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 660917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 733708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 658333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2704395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 998416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1578750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 248708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d14426c0a6162e645b04f051ecc1da73be6fc872",
          "message": "Move MTL tests and add a few (#491)",
          "timestamp": "2024-12-10T10:29:24+01:00",
          "tree_id": "3be9055df0349510959b853d593e33986f5a52e9",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/d14426c0a6162e645b04f051ecc1da73be6fc872"
        },
        "date": 1733825333492,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27132,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 793708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 556167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 593875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 766209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 610667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 565084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 824875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 825938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1344562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1389833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2093583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1830874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1689375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1661708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3921666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3195958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1765125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1349500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1035375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 660583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1044687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 660917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2716417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1015292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1576791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 590167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5133928042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6740222625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1147095896.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 711042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1631916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8785708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1557875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2693167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 490416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 478416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 464917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 789083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 622041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 561208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 570167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 766459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 607750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 86000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 78083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1355875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1395104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1817917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1572666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1385542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1361167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 158250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2966688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1444000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1362146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 727709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 664750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 742479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12224\nallocs=469\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 657917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2713250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1013333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1627375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 252375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c2207d2e5027534f42992d380dd3af518b6a44e8",
          "message": "Fix copy tests (#493)",
          "timestamp": "2024-12-10T16:15:54-04:00",
          "tree_id": "2c97b9c48daa979597a062bb043bf8a6027849c1",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/c2207d2e5027534f42992d380dd3af518b6a44e8"
        },
        "date": 1733864235105,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26645.85714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 908709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 592333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 551375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 551146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 922187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 787791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 549167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 646917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 624625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1427083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1516125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2288375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2023792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1781375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1755687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2578834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3571041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1896063,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1417541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 954583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 710270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 952271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 704396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 3168041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1074958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1885750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 816417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5220350875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6674789459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1170613792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 748083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1676958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 21394667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1658146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2814542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 459916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 462041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26336.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 463875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 914542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 591792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 549291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 551229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 816416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 788479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 81625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 82125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 83250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1428208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1513750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2080562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1745584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1506875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1449500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 157584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3297083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1557791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1452916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 726917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 698979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 694146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 699833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 3037396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1093083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1876083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 209542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "distinct": true,
          "id": "634cec7a042ee34d919c5a32122404151261c238",
          "message": "Move benchmark results into a subfolder.\n\n[skip tests]",
          "timestamp": "2024-12-11T14:06:50+01:00",
          "tree_id": "7eedcbed3061321e656119040321359e57346243",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/634cec7a042ee34d919c5a32122404151261c238"
        },
        "date": 1733923559959,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 865083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 548542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 584292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 756500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 605542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 658041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 701083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 824854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1325958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1382792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2067791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1825375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1674750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1646437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3884646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3164458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1740125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1346458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1020729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 664250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1032125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 659542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2720542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10816\nallocs=314\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1011208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1574854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 557250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5138248250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6754936625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1151697916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 719667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1560604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10389833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1566416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2542084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 487187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 469791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26607.14285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 453208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 790125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 668750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 573042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 596500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 780209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 618625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 98084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1330458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1388291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1768750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1573208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1392333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1363479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 152500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2956500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1459687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1350209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 718916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 670875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 722583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 669292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2722021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10816\nallocs=314\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1005625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1573479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 248062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8c119cfc518fa5b90da8b9e6b9eb782d4d79f7c6",
          "message": "Fix global linear indexing (`fill!`) (#496)",
          "timestamp": "2024-12-16T09:18:12+01:00",
          "tree_id": "d531405e0dbdee1a063c1d58138802e782ca9f38",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/8c119cfc518fa5b90da8b9e6b9eb782d4d79f7c6"
        },
        "date": 1734339479641,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27535.714285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 462667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 801500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 649125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 556667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 589000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 757125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 612979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 636833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 802250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 811416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1335417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1385542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2069208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1807625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1689417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1654417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3888771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3174145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1740875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1351417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1020500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 658166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1042750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 656958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2558833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1022709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1587645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 591375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5143804334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6880237708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1161084541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 715041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5208\nallocs=206\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1632375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10058875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1585312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2691375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8608\nallocs=343\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 462625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 460479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10104.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26138.916666666664,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 468542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 809834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 664917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 556875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 595666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 787624.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 602166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 88708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 79167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1355375.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1387104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1754499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1593625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1384875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1353459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 160834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2942875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1457563,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16024\nallocs=577\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1364667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 726000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 654625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 736917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 664458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2550125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1026542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1599229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 237042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5056e3348749686c452b91dcd1ea541ea87599a1",
          "message": "Initial support for MPSNDArray (#499)",
          "timestamp": "2024-12-16T17:33:46+01:00",
          "tree_id": "67e6f94bb26330cb72160408eeb5e0a46220deb1",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/5056e3348749686c452b91dcd1ea541ea87599a1"
        },
        "date": 1734369581808,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 25920.083333333336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 464792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 811812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 673333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 553250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 601000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 763896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 616375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 660854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 619854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 834687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1328542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1387334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2070500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1824916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1682146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1637959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3891833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3177708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1740229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1343375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1035708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 651667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1037167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 657917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2537625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1022708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1577291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 621750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5243893542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6538101604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1165440583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 705833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1588833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10079959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1568145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2643625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 444959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 446541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10666.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 15167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25503.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 475125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 752770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 655458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 560916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 598459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 774208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 629583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 85500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 91792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1347021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1383874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1783167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1585458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39616\nallocs=1447\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1392020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1354916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 162042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2968958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1461729.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1364792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 730458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 656958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 746250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 660562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2528541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1025917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1580896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 242437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "distinct": true,
          "id": "5a8daaa3d5b31b8626e6f553953f1c2741b05d84",
          "message": "Benchmarks: Fix workflow.\n\n[only benchmarks]",
          "timestamp": "2024-12-18T10:17:09+01:00",
          "tree_id": "5cc1c4b3c11a237474291503453ba6fc55652e0c",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/5a8daaa3d5b31b8626e6f553953f1c2741b05d84"
        },
        "date": 1734514670731,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26673.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 458208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 852625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 555625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 589208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 802125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 595625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 679834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 666958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 518021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1339437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1374708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2065062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45688\nallocs=1653\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1816354,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1690417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1658583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3599687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3169854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1746000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1351313,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1041021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 655041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1040833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 660979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2523625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1020958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1584104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 580395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5246987875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6666328874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1165891041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 718542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1537583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8342167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1552229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2618584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 458000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 473584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10069.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 15479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26243.083333333336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 473062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 806375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 662875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 555333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 589000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 737791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 636875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 89084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 84916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1339729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1387291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1813291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1572375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39616\nallocs=1447\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1385750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1363396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 158084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2988208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65656\nallocs=2411\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1467875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1368229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 731459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 669917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 733042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 671958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2426458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1015708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1591625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 250687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "84447c4db5b2701f3618900a947b3f90adb53bf9",
          "message": "Fix `MPSNDArrayDescriptor` wrapper (#502)\n\nDon't reverse dimensions automatically",
          "timestamp": "2024-12-19T10:42:27+01:00",
          "tree_id": "8bd198c3d210ed12a28014c2d2442d407729da42",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/84447c4db5b2701f3618900a947b3f90adb53bf9"
        },
        "date": 1734603700312,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26538.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 455500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 810750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 641541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 547500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 581708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 754708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 576250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 678208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 821375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 674333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1347708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1384500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8864\nallocs=326\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2102291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1807917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1692020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1656666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3920667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3200708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1765458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1340875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1037000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 666542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1027813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 712062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2564583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1019958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1593584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 541583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5246979375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6653773250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1164822625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 713667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1535999.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9724167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1540854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2638000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 476334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 477542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 27142.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 461000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 817792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 636042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 542833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 577542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 775874.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 573604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 88375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 85375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 84084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1353166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1386916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8864\nallocs=326\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1831541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1577750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1387770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1340584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 158625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2945562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1456666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1357833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 730167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 669417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 743312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 667000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2570541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1026187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1596125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 245812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ea1d6ad8ace75f25a77605a9dc6437ba1512cdb1",
          "message": "Generate MTL and MPS structs and enums with Clang.jl (#492)\n\n* Add Docs and wrapping readme\n\nCo-authored-by: Tim Besard <tim.besard@gmail.com>\n\n---------\n\nCo-authored-by: Tim Besard <tim.besard@gmail.com>",
          "timestamp": "2024-12-20T10:01:23-04:00",
          "tree_id": "e7fc5926ccc2d5229244cb939745af76bc6d2254",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/ea1d6ad8ace75f25a77605a9dc6437ba1512cdb1"
        },
        "date": 1734705832809,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27217.214285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 464500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 826000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 572854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 598542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 771187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 585687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 662042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 683709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 804500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1317083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1373541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2028709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1812688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1707834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1660229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3568021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3163792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1739750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1349604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1035312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 654666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1034083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 661125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2484604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1024083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1571500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 577000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5769911666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6647448292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1167766604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 719229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1521583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9443084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1487604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2653771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 531041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 472333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 10201.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 13917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26330.357142857145,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 476083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 768750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 657000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 554959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 599625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 735208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 626959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 87125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 82292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1329375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1383292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1790458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1556709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1376958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1355917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 152208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2949792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1454250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1354125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21656\nallocs=717\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 728312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 662854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 726833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 657667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2555792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1009333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1585666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 248500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e762b0160d4f666531d74e17d96646cc4b2e9f56",
          "message": "Adjust random tests to avoid spurious test failures. (#506)",
          "timestamp": "2024-12-21T08:32:32+01:00",
          "tree_id": "c2f99e96b7faf2827821765846943e0a4d090332",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/e762b0160d4f666531d74e17d96646cc4b2e9f56"
        },
        "date": 1734768929203,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26583.333333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 461750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 794917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 643041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 553833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 599792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 764125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 608917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 679770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 674500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 756021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1333604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1393875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2044667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1835791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1694083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1666562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3573771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3178292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1777459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1351583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1033562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 672104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1031375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 667750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2567312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1035667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1607458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 584667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5772774167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6734636291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1170506875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 708167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1570895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9804542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1559062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2645000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 457583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 459812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 9521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 27125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 469875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 824500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 625166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 555834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 588000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 804125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 586542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 86917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 88542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 76875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1344375.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1388042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1813583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1563708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1415500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1367312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 157542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2948229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1471291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1365354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 721687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 671750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 740270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 672271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2578708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1018083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1601458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 245562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "182095779fe4fb6af497a1824a09383dbc5de3cf",
          "message": "Bump LLVM downgrader (#507)\n\nv0.6 includes fixes for constant values and known intrinsics.",
          "timestamp": "2024-12-21T22:25:25+01:00",
          "tree_id": "6acb60b9adc9af28ac2e19c143fd539ae7dfc046",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/182095779fe4fb6af497a1824a09383dbc5de3cf"
        },
        "date": 1734818840310,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26795.083333333336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 465417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 766458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 658791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 565416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 585541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 794583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 603208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 661083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 821334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 769291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1343959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1393208.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2103125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1841854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1694416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1667875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3979292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3188666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1761458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1362292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1043000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 663833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1059834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 662417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2564437,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1031750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1593354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 591521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5763160083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6663954270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1169291583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 714333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1647416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8868125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1566291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2727167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 457375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 456500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 37291.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 27152.833333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 472750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 791708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 659791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 552083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 595834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 782875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 578000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 86667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 89875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 78083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1354292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1402292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1845625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45504\nallocs=1639\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1610291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1395270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1363083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 158584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2985875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65656\nallocs=2411\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1464479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1373417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 739562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 671771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 751584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 668417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2553000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1035875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1593333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 251771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6a760a60a33b864b2c7fa3e856e38e1e6299f871",
          "message": "Remove type piracy and enable function wrapping (#505)\n\nIgnore MPS functions for now since MPS does not seem to have a dylib.",
          "timestamp": "2024-12-24T11:01:48+01:00",
          "tree_id": "1ae235dea8ea251d4fca9dde25c2087061fb1e24",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/6a760a60a33b864b2c7fa3e856e38e1e6299f871"
        },
        "date": 1735037260267,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27270.785714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 460209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 804875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 646333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 548916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 587375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 716083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 615791.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 677125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 640688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 817625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1329687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1382229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2073708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1799041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1688292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1650020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3252542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3147375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1736042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1348917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1029542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 650292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1025917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 657229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2553708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1027750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1585916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 580417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5847134584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6545482667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1169724375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 713125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1580770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 9774042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1598000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2571895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 457542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 458645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25166.714285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 469917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 825666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 616625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 547708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 591458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 734166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 610312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 88000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 86291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1336542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1384250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1765020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44648\nallocs=1608\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1558812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38760\nallocs=1416\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1396125.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1360145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 153334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2940000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64152\nallocs=2363\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1442291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1358562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 723750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 654792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 728687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 660125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2522521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1019375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1579917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 233417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cf21f9db92e7359105bd0ca44727dfa74739c56d",
          "message": "Update Documenter (#515)",
          "timestamp": "2025-01-04T14:51:16-04:00",
          "tree_id": "e3c2e52daa6b488baa1d613c1f637ae2d16e7bed",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/cf21f9db92e7359105bd0ca44727dfa74739c56d"
        },
        "date": 1736019452425,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27517.85714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 457584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 821104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 616500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 548167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 582125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 764500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 598542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 646125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 810041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 644937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1320000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1378958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2020834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45656\nallocs=1651\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1779625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39768\nallocs=1459\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1696916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1650188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=18152\nallocs=675\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3549500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3174729.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=66504\nallocs=2453\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1747792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16888\nallocs=618\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1339417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1031312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 656042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1027125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=13056\nallocs=510\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 666791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2554792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1026291.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1591209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 595583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5856015937.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 6607073479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1171437917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 699417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1594083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5536\nallocs=219\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 11850083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11808\nallocs=466\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1622417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5048\nallocs=205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2525791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8448\nallocs=342\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 466979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3384\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 479562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3392\nallocs=137\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 7958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=896\nallocs=32\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 14834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 466145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3504\nallocs=132\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 784895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 648729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 543291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 579958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 764687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 636958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 87791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 94417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 83333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1327875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=33488\nallocs=1233\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1378167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8704\nallocs=325\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1773979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=45432\nallocs=1632\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1575646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39616\nallocs=1447\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1391812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1352062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17128\nallocs=633\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 155709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2959083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65496\nallocs=2410\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1459687.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15864\nallocs=576\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1353292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21496\nallocs=716\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 727896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 660583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 730833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12064\nallocs=468\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 662708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6352\nallocs=231\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2539083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10656\nallocs=313\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1019208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9024\nallocs=310\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1600708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9704\nallocs=312\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 238729.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "de8eed59b79b19cbbcbb1053c16dea5c2aa49667",
          "message": "Remove `device_code_agx` (#512)\n\nIt doesn't work on M3 anyway, and the Python\r\ndependency is quite heavy.",
          "timestamp": "2025-01-06T08:37:52+01:00",
          "tree_id": "4250ae495043ca18f93ae011766adbb42ca0cfd7",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/de8eed59b79b19cbbcbb1053c16dea5c2aa49667"
        },
        "date": 1736151729946,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26726.14285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 467625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 925917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 593500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 550042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 558625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 858229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 831083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 548209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 688458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 626000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1426625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1524853.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2284896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44952\nallocs=1616\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2055500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39160\nallocs=1429\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1826500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1739770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2446687,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3543000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65384\nallocs=2397\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1920312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16568\nallocs=602\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1433354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 929542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 715208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1000417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 707604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2685542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1095645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1858833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 839083,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5818530541.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 3076733499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1161722437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 769000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1682812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5456\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 20107416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11728\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1668458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2862250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8368\nallocs=338\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 465667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3304\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 471958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3312\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 15000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26149.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 466375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 917458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 595833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 550375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 558416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 855708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 809708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 81250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 82417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 82125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1417292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1506208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2042854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=43928\nallocs=1572\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1762250.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38976\nallocs=1415\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1513750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1440125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 160334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3301709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64376\nallocs=2354\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1580791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15544\nallocs=560\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1457833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 692271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 700958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 718458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 717104.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2679896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1119959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1852000.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 210584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6ac7f3c349202e87ddc55525aa30f6786c781970",
          "message": "Fix typo in random tests (#514)",
          "timestamp": "2025-01-06T12:39:35+01:00",
          "tree_id": "a89e5c479a2ed58acbb174b5aa6d7708df29bb9f",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/6ac7f3c349202e87ddc55525aa30f6786c781970"
        },
        "date": 1736166020775,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26131,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 463417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 896416,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 591354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 550958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 550458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 892521,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 776917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 594083.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 658708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 649209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1443834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1500708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2321979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44936\nallocs=1615\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2055750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39128\nallocs=1427\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1811916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1734959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2445625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3496146.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65384\nallocs=2397\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1932458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16568\nallocs=602\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1457562.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 970146,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 729750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 985166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 707166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2696500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1122916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1861125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 809834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5830401479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 3083600896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1160891875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 760875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1682042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5456\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 19850395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11728\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1666916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2824958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8368\nallocs=338\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 460854.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3304\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 468479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3312\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 8250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 15042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26416.666666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 463291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 933042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 589020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 551208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 559708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 909896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 823646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 81542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 82042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 79958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1441770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1534062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2050459,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44784\nallocs=1603\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1778792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38120\nallocs=1384\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1511104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1457771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 161334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3295666.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64376\nallocs=2354\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1575500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15544\nallocs=560\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1465729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 724292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 704958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 719478.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 710875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2675917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1127021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1878708.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 209292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "28689358+christiangnrd@users.noreply.github.com",
            "name": "Christian Guinard",
            "username": "christiangnrd"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aae82e4a223a87a24b78925fec2c461af25dbf08",
          "message": "Test loading of package on unsupported platforms (#509)\n\nCo-authored-by: Tim Besard <tim.besard@gmail.com>",
          "timestamp": "2025-01-07T09:05:33+01:00",
          "tree_id": "6f6668750f0f18a8781e4d65ab241f530a7d56df",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/aae82e4a223a87a24b78925fec2c461af25dbf08"
        },
        "date": 1736240104536,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26369,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 458729.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 901583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 593625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 546708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 556375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 873250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 784791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 547979.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 596917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 676354.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1435750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1496709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2348958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44936\nallocs=1615\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 2035396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39128\nallocs=1427\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1828250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1752000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 2406458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3509708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65384\nallocs=2397\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1865583.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16568\nallocs=602\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1446125.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 974896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 696542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 902166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 701167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2665417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1118875.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1854333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 783917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5826737333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 3079494895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1160708708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 765458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1661833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5456\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 19882209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11728\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1644041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2825375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8368\nallocs=338\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 463333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3304\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 462792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3312\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 9507,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 26387,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 463084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 912646,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 594271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 554500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 544875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 888209,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 795791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 83750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 83750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 82375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1418187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1503708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 2140583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44784\nallocs=1603\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1760916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38120\nallocs=1384\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1519000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1438958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 159542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 3245229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64376\nallocs=2354\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1601437.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15544\nallocs=560\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1441500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 702250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 701770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 714541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 691500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2675333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1075625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1806250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 212292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dff150fda84839a0e989183f5d1cff2ed81d8b48",
          "message": "Bump version.",
          "timestamp": "2025-01-08T11:20:19+01:00",
          "tree_id": "900b2f0f751c959a807d10eb04930865ae2e516d",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/dff150fda84839a0e989183f5d1cff2ed81d8b48"
        },
        "date": 1736334025024,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 26226.285714285714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 463709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 790792,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 630145.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 574917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 599125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 757291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 619812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 646417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 832395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 795770.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1335417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1385625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2064084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44936\nallocs=1615\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1822062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39128\nallocs=1427\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1693166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1657000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3953312.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3150667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=63976\nallocs=2346\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1753708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16568\nallocs=602\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1346916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1028416.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 649166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1045479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 654875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2558812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1018166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1585166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 546125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5746549250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 3031731729.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1141085375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 702604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1638604.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5456\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 10832625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11728\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1552542,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2686333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8368\nallocs=338\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 458104,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3304\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 461500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3312\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 7875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 15000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 25604.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 460250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 788750.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 640229.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 576250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 589020.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 755041.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 651188,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 88125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 78000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 77417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1335500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1385917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1819271,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44216\nallocs=1573\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1569270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38152\nallocs=1386\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1385708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1362250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 158916,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2945583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=62984\nallocs=2304\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1455750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15544\nallocs=560\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1362583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 725625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 663375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 732812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 654875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2540667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1026375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1581750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 251584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "tim.besard@gmail.com",
            "name": "Tim Besard",
            "username": "maleadt"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6efc3491ed501ddf2d1ec99eafe56dcd6c1a8d2c",
          "message": "Add Runic action to suggest formatting changes. (#517)",
          "timestamp": "2025-01-08T12:39:18+01:00",
          "tree_id": "26c4b2e2525edeacb05955d0a80e6e8405cc2a8f",
          "url": "https://github.com/JuliaGPU/Metal.jl/commit/6efc3491ed501ddf2d1ec99eafe56dcd6c1a8d2c"
        },
        "date": 1736339112663,
        "tool": "julia",
        "benches": [
          {
            "name": "private array/construct",
            "value": 27023.714285714286,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/broadcast",
            "value": 460417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn/Float32",
            "value": 856604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/randn!/Float32",
            "value": 624084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Int64",
            "value": 562958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand!/Float32",
            "value": 601958.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Int64",
            "value": 767167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/random/rand/Float32",
            "value": 619854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_gpu",
            "value": 652833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=832\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/cpu_to_gpu",
            "value": 680291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copyto!/gpu_to_cpu",
            "value": 837667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=56\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/1d",
            "value": 1323854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/accumulate/2d",
            "value": 1390021,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/int",
            "value": 2091375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=44936\nallocs=1615\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findall/bool",
            "value": 1825917,
            "unit": "ns",
            "extra": "gctime=0\nmemory=39128\nallocs=1427\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/int",
            "value": 1683500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findfirst/bool",
            "value": 1641667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17832\nallocs=659\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/scalar",
            "value": 3840167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12272\nallocs=533\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/logical",
            "value": 3174604,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64264\nallocs=2347\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/1d",
            "value": 1744896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16568\nallocs=602\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/iteration/findmin/2d",
            "value": 1338500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/1d",
            "value": 1028375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/reduce/2d",
            "value": 651667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/1d",
            "value": 1015000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=12736\nallocs=494\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/reductions/mapreduce/2d",
            "value": 651875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/4d",
            "value": 2489500,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/2d",
            "value": 1023479.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/permutedims/3d",
            "value": 1582916.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "private array/copy",
            "value": 594500.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2288\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/precompile",
            "value": 5747134375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/ttfp",
            "value": 3037577833.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":60,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "latency/import",
            "value": 1141637625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1360\nallocs=46\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":30,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/metaldevrt",
            "value": 709666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=1",
            "value": 1579791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5456\nallocs=215\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=3",
            "value": 8864667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11728\nallocs=462\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/reference",
            "value": 1608125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4968\nallocs=201\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "integration/byval/slices=2",
            "value": 2584187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8368\nallocs=338\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing",
            "value": 454625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3304\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/indexing_checked",
            "value": 451458.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3312\nallocs=133\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "kernel/launch",
            "value": 7791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816\nallocs=28\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/stream",
            "value": 14208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=128\nallocs=5\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "metal/synchronization/context",
            "value": 15125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=272\nallocs=13\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/construct",
            "value": 27000,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1056\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/broadcast",
            "value": 459042,
            "unit": "ns",
            "extra": "gctime=0\nmemory=3424\nallocs=128\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn/Float32",
            "value": 796084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/randn!/Float32",
            "value": 623250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Int64",
            "value": 561791,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand!/Float32",
            "value": 583250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=432\nallocs=19\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Int64",
            "value": 772709,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1832\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/random/rand/Float32",
            "value": 623187.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1816\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_gpu",
            "value": 81958,
            "unit": "ns",
            "extra": "gctime=0\nmemory=864\nallocs=40\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/cpu_to_gpu",
            "value": 82167,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copyto!/gpu_to_cpu",
            "value": 81834,
            "unit": "ns",
            "extra": "gctime=0\nmemory=528\nallocs=23\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/1d",
            "value": 1345479,
            "unit": "ns",
            "extra": "gctime=0\nmemory=32928\nallocs=1205\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/accumulate/2d",
            "value": 1383583,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8544\nallocs=317\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/int",
            "value": 1795062.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=43928\nallocs=1572\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findall/bool",
            "value": 1579125.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=38120\nallocs=1384\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/int",
            "value": 1378125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findfirst/bool",
            "value": 1345250,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16808\nallocs=617\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/scalar",
            "value": 151895.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=113\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/logical",
            "value": 2975708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=62968\nallocs=2303\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/1d",
            "value": 1448729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15544\nallocs=560\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/iteration/findmin/2d",
            "value": 1359645.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=21176\nallocs=700\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/1d",
            "value": 723292,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/reduce/2d",
            "value": 654667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/1d",
            "value": 734333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=11744\nallocs=452\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/reductions/mapreduce/2d",
            "value": 650750,
            "unit": "ns",
            "extra": "gctime=0\nmemory=6192\nallocs=223\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/4d",
            "value": 2539541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=10576\nallocs=309\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/2d",
            "value": 1019812.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8944\nallocs=306\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/permutedims/3d",
            "value": 1579395.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=9624\nallocs=308\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "shared array/copy",
            "value": 240000.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2320\nallocs=74\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}