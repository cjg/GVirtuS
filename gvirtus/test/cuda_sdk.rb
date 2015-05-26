#!/usr/bin/env ruby -w

tests = [ "3dfd",
  "alignedTypes",
  "asyncAPI",
  "bandwidthTest",
  "binomialOptions",
  "BlackScholes",
  "clock",
#  "convolutionFFT2D",
  "convolutionSeparable",
  "convolutionTexture",
  "cppIntegration",
  "dct8x8",
  "deviceQuery",
  "deviceQueryDrv",
  "dwtHaar1D",
  "dxtc",
  "eigenvalues",
  "fastWalshTransform",
  "histogram",
  "lineOfSight",
  "matrixMul",
  "matrixMulDynlinkJIT",
  "memcpyTest",
  "MersenneTwister",
  "MonteCarlo",
  "ptxjit",
  "quasirandomGenerator",
  "radixSort",
  "recursiveGaussian",
  "reduction",
  "scalarProd",
  "scan",
  "scanLargeArray",
  "simpleAtomicIntrinsics",
  "simpleCUBLAS",
  "simpleCUFFT",
  "simplePitchLinearTexture",
  "simpleStreams",
  "simpleTemplates",
  "simpleTexture",
  "simpleVoteIntrinsics",
  "simpleZeroCopy",
  "SobolQRNG",
  "sortingNetworks",
  "template",
  "threadFenceReduction",
  "threadMigration",
  "transpose",
  "transposeNew"]

lib_path = "#{ARGV[0]}:#{ENV['LD_LIBRARY_PATH']}"
sdk_path = ARGV[1]

ENV['LD_LIBRARY_PATH'] = lib_path

log = File.new("cuda_sdk.log", "w");
failed = []
tests.each do |t|
  $stdout.write("Running '#{t}' ... ")
  $stdout.flush
  out = `echo | #{sdk_path}/C/bin/linux/release/#{t} 2>&1`
  if $? != 0 or out =~ /FAILED/
    puts "FAILED"
    puts out
    failed << t
  else
    puts "PASSED"
  end
  log.write("#{'*'*80}\n#{t}\n#{'*'*80}\n#{out}\n#{'*'*80}\n\n")
end

log.close

puts
puts "*"*30
puts "Executed tests: #{tests.length}"
puts "  Passed      : #{tests.length - failed.length}"
puts "  Failed      : #{failed.length}"
puts "Failed tests  : #{failed.join(', ')}" if failed.length > 0
puts "*"*30

