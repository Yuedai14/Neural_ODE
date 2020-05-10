package gemmini

import chisel3._
import chisel3.iotesters._
import org.scalatest.{FlatSpec, Matchers}
import TestUtils._
import scala.util.Random

// line buffer tester
class LineBufferTester(c: LineBufferUnit[SInt, UInt], inputSeq: Seq[Seq[Int]], kernel: Seq[Seq[Int]], inSize: Int, realKernelSize: Int, continuous: Boolean, goldResult: Seq[Seq[Int]], verbose: Boolean = false) extends PeekPokeTester(c)
{
	println("Begin test")
	val maxWait = 100
	var totalcycle = 0
	poke(c.io.out.ready, true)
	poke(c.io.lb_control, false)
	// set configuration
	val k_len = kernel.length
	poke(c.io.k_len.valid, true)
	poke(c.io.k_len.bits, k_len)
	
	poke(c.io.lb_config.valid, true)
	poke(c.io.lb_config.bits.inSize, inSize)
	poke(c.io.lb_config.bits.realKernelSize, realKernelSize)
	poke(c.io.lb_config.bits.continuous, continuous)
	poke(c.io.lb_control, true)

	step(1)
	poke(c.io.k_len.valid, false)
	poke(c.io.lb_config.valid, false)
	
	// set kernels
	println("Setting kernel")
	var cycleWaiting = 0
	while(peek(c.io.k.ready) == BigInt(0) && cycleWaiting < maxWait){
		cycleWaiting += 1
		if (cycleWaiting >= maxWait){expect(false, "wait for k too long")}
		step(1)
		totalcycle += 1
	}
	for(in_k <- kernel){
		poke(c.io.k.valid, true)
		(in_k zip c.io.k.bits).foreach{case(k, in) => poke(in(0), k)}
		step(1)
		totalcycle += 1
	}
	poke(c.io.k.valid, false)

	// feed inputs and check result
	cycleWaiting = 0
	while(peek(c.io.b.ready) == BigInt(0) && cycleWaiting < maxWait){
		cycleWaiting += 1
		if (cycleWaiting >= maxWait){expect(false, "wait for input too long")}
		step(1)
		totalcycle += 1
	}
	println("Begin to feed input")
	val cols = inputSeq.length
	val samples = inputSeq(0).length
	var goldResultPtr = 0
	for(i <- 0 until samples){
		poke(c.io.b.valid, true)
		c.io.b.bits.zipWithIndex.foreach{case(b, idx) => poke(b(0), inputSeq(idx)(i))}

		if(peek(c.io.out.valid) == BigInt(1)){
			c.io.out.bits.zipWithIndex.foreach{case(out, idx) => expect(out(0), goldResult(idx)(goldResultPtr))}
			goldResultPtr += 1
		}
		
		step(1)
		totalcycle += 1
	}
	
	cycleWaiting = 0
	while(goldResultPtr < samples && cycleWaiting < maxWait){
		if(peek(c.io.out.valid) == BigInt(1)){
			c.io.out.bits.zipWithIndex.foreach{case(out, idx) => expect(out(0), goldResult(idx)(goldResultPtr))}
			goldResultPtr += 1
		}
		cycleWaiting +=1
		step(1)
 		totalcycle += 1
	}
	
	println("Finish check!")
	print(totalcycle)
}

// Convience function for running tester
object SIntLineBufferTester{
	def apply(meshRows: Int, meshColumns: Int, kernelSize: Int, inputBufferSize: Int, padding: Int, inputSeq: Seq[Seq[Int]], kernel: Seq[Seq[Int]], inSize: Int, realKernelSize: Int, continuous: Boolean, goldResult: Seq[Seq[Int]]): Boolean = {
		iotesters.Driver.execute(Array("--backend-name", "treadle", "--generate-vcd-output", "on"), () => new LineBufferUnit(SInt(16.W), SInt(16.W), SInt(16.W), UInt(32.W), 1, 1, meshRows, meshColumns, kernelSize, inputBufferSize, padding)) {
			c => new LineBufferTester(c, inputSeq, kernel, inSize, realKernelSize ,continuous, goldResult)
		}
	}
} 

class LineBufferSpec extends FlatSpec with Matchers{
	behavior of "LineBufferUnit"
	val meshColumns = 32
	val meshRows = 16
	val kernelSize = 3
	val inputWeightSize = 3
	val padding = 1
	val inputChannels = 4

	val inputWeights = Array.ofDim[Int](meshColumns, inputChannels, inputWeightSize, inputWeightSize)
	val kernel = Array.ofDim[Int](kernelSize, kernelSize)
	
	for(col <- 0 until meshColumns){
		for(channel <- 0 until inputChannels){
			for(weightrow <- 0 until inputWeightSize){
				for(ele <- 0 until inputWeightSize){
					inputWeights(col)(channel)(weightrow)(ele) = Random.nextInt(10)
				}
			}
		}
	}

	for(rows <- 0 until kernelSize){
		for(ele <- 0 until kernelSize){
			kernel(rows)(ele) = Random.nextInt(10)
		}
	}
	// add padding 
	val padded_inputWeight = Array.ofDim[Int](meshColumns, inputChannels, (inputWeightSize+2*padding)*(inputWeightSize+2*padding))
    val zeroSeq = Array.fill[Int](2*padding+inputWeightSize)(0)
	val zeroSingleSeq = Array.fill[Int](padding)(0)	
	for(i <- 0 until meshColumns){
		for(k <- 0 until inputChannels){
    		var padded_weight_flatten = zeroSeq
			for(m <- 0 until inputWeightSize){
				if(m == inputWeightSize - 1){
					padded_weight_flatten = padded_weight_flatten ++ zeroSingleSeq ++ inputWeights(i)(k)(m) ++ zeroSingleSeq ++ zeroSeq
				}
				else{
					padded_weight_flatten = padded_weight_flatten ++ zeroSingleSeq ++ inputWeights(i)(k)(m) ++ zeroSingleSeq
				}
			}
			padded_inputWeight(i)(k) = padded_weight_flatten
			//println(padded_weight_flatten.mkString(" "))
		}
	}

	// Calculate goldenResult
	val kernel_flatten = kernel.flatten
	val goldResult_raw = Array.ofDim[Int](meshColumns, inputChannels, inputWeightSize*inputWeightSize)
	for(i <- 0 until meshColumns){
		for(k <- 0 until inputChannels){
			var kernelPtr = 0
			val slide_window = (kernelSize-1)*(inputWeightSize+2*padding) + kernelSize
			for(m <- 0 until (inputWeightSize * inputWeightSize)){
				val calculate_array = padded_inputWeight(i)(k).slice(kernelPtr, kernelPtr+slide_window).sliding(kernelSize, inputWeightSize+2*padding).toArray.flatten

				goldResult_raw(i)(k)(m) = (kernel_flatten,calculate_array).zipped.map(_*_).reduce(_+_)
				if((kernelPtr+1)%(inputWeightSize+2*padding) == inputWeightSize){
					kernelPtr += 2*padding + 1
				}else{
					kernelPtr += 1
				}
			}
		}
	}	

	// transform kernel
	val zeroSeq_num = meshColumns - (kernelSize*kernelSize) % meshColumns
	val zeroPaddingKernel = Array.fill[Int](zeroSeq_num)(0)
	val zeroPaddedKernel = (kernel.flatten ++ zeroPaddingKernel).toSeq.sliding(meshColumns, meshColumns).toSeq

	println("kernel:")
	kernel.foreach((a) => println(a.mkString(" ")))
	println("zero padded kernel")
	zeroPaddedKernel.foreach((a) => println(a.mkString(" ")))
	println("input Weights:")
	inputWeights.foreach(_.foreach{(a) => println("new in:")
										  a.foreach((m) => println(m.mkString(" ")))											
	})

	val inputSeq = inputWeights.map(_.map(_.flatten.toSeq).flatten.toSeq)
	println("inputSeq")
	inputSeq.foreach((a) => println(a.mkString(" ")))
	val goldResult = goldResult_raw.map(_.flatten.toSeq)
	println("goldenRes")
	goldResult.foreach((a) => println(a.mkString(" ")))
   	// tag
   	/*val tag = new Bundle with TagQueueTag{
		val rob_id = UDVlid(UInt(log2Up(16).W))
		val addr = UInt(8.W)
		val rows = UInt(9.W)
		val cols = UInt(9.W)

		override def make_this_garbage(dummy: Int = 0): Unit = {
			rob_id.valid := false.B
		}
	} */

	"LineBuferTester" should "work" in{
		SIntLineBufferTester(meshRows, meshColumns, 4, 64, padding, inputSeq, zeroPaddedKernel, inputWeightSize, kernelSize, true ,goldResult) should be (true)
	}


} 



