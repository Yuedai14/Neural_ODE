package gemmini

import chisel3._
import chisel3.util._

import gemmini.Util._


// TODO" This is a naive implementation of vector multiplication
class VecMul[T <: Data](inputType: T, outputType: T, vecSize: Int)(implicit ev: Arithmetic[T]) extends Module{
	import ev._
    val io = IO(new Bundle{
		val in_vec_a = Input(Vec(vecSize, inputType))
		val in_vec_b = Input(Vec(vecSize, inputType))
		
		val out_c = Output(outputType)
	})
	
	io.out_c := (io.in_vec_a, io.in_vec_b).zipped.map(_*_).reduce(_+_)
}

class LineBufferConfig[T <: Data : Arithmetic](inputSize: Int, kernelSize: Int) extends Bundle{
	val inSize = Input(UInt(log2Up(inputSize).W))
	val realKernelSize = Input(UInt(log2Up(kernelSize).W))
	val continuous = Input(Bool())

	override def cloneType: LineBufferConfig.this.type = new LineBufferConfig(inputSize, kernelSize).asInstanceOf[this.type]
}

class LineBuffer[T <: Data : Arithmetic, U <: /*TagQueueTag with*/ Data]
  (inputType: T, val outputType: T, accType: T,
   tagType: U, tileRows: Int, tileColumns: Int, meshRows: Int, meshColumns: Int,
   kernelSize: Int = 4, inputBufferSize:Int = 64, padding:Int = 1)
  extends Module {

  val B_TYPE = Vec(meshColumns, Vec(tileColumns, inputType))
  val K_TYPE = Vec(meshColumns, Vec(tileColumns, inputType))
  val MAX_SUPPORT_INPUTSIZE = inputBufferSize/kernelSize


  val io = IO(new Bundle {
    val b = Flipped(Decoupled(B_TYPE))
    val k = Flipped(Decoupled(K_TYPE))
	val k_len = Flipped(Decoupled(UInt((log2Up(kernelSize*kernelSize/meshColumns)+1).W)))
    // LineBuffer control, only work when this signal is high
	// Execute controller needs to count output numbers and set lb_control low after finishing all convolutions
    val lb_control = Input(Bool())
	val lb_config = Flipped(Decoupled(new LineBufferConfig(MAX_SUPPORT_INPUTSIZE, kernelSize)))
	
    val tag_in = Flipped(Decoupled(tagType))
    val tag_out = Output(tagType)

    val out = Decoupled(B_TYPE) // TODO make this ready-valid

  })

  val kernelMatrix = RegInit(VecInit(Seq.fill(kernelSize*kernelSize)((0.U).asTypeOf(inputType))))
  // This is line buffer with padding
  val inputBuffer =  RegInit(VecInit(Seq.fill(meshColumns)(VecInit(Seq.fill(inputBufferSize)((0.U).asTypeOf(inputType)))))) 
  
  // pointer for line buffer
  val bufferPointer_x = RegInit((0.U)((2*log2Up(inputBufferSize)).W))
  val bufferPointer_y = RegInit((0.U)((2*log2Up(inputBufferSize)).W))
  val bufferPointer   = Wire(UInt((2*log2Up(inputBufferSize)).W))
  val kernelPointer_x = RegInit((0.U)((2*log2Up(inputBufferSize)).W))
  val kernelPointer_y = RegInit((0.U)((2*log2Up(inputBufferSize)).W))
  val kernelPointer   = Wire(UInt((2*log2Up(inputBufferSize)).W))
 

  val lb_control_prev = RegInit(false.B)
  val kernelMoveBegin = RegInit(false.B)
  val kernelWritten = RegInit(false.B)
  val configWritten = RegInit(false.B)
  val convSeqWritten = RegInit(false.B)
  
  // Tag regitser and store current tag
  val tagReg = Reg(tagType.cloneType)
  val tag_written = RegInit(false.B)
  when(io.tag_in.fire()){
  	tagReg := io.tag_in.bits
	tag_written := true.B
  }
  io.tag_in.ready := !tag_written
  
  // Store previous lb_contrl_prev
  lb_control_prev := io.lb_control
  
  // Set up config registers
  val inSize = RegInit(0.U(log2Up(MAX_SUPPORT_INPUTSIZE).W))
  val realKernelSize = RegInit(0.U(log2Up(kernelSize).W))
  val continuous = RegInit(true.B)

  when(io.lb_config.fire()){
	inSize := io.lb_config.bits.inSize
	realKernelSize := io.lb_config.bits.realKernelSize
	continuous := io.lb_config.bits.continuous
	bufferPointer_x := padding.U
	bufferPointer_y := padding.U
	configWritten := true.B
  }
  bufferPointer := bufferPointer_y * inSize + bufferPointer_x 
  kernelPointer := kernelPointer_y * inSize + kernelPointer_x

  // Calculate convSeq
  val convSeq = Reg(Vec(kernelSize*kernelSize, UInt((log2Up(inputBufferSize)+1).W)))
  val convSeqCounter_x = RegInit(0.U((log2Up(kernelSize*kernelSize).W)))
  val convSeqCounter_y = RegInit(0.U((log2Up(kernelSize*kernelSize).W)))
  val convSeqCounter   = Wire(UInt((log2Up(kernelSize*kernelSize).W)))
  convSeqCounter := convSeqCounter_y * realKernelSize + convSeqCounter_x
  when(configWritten && !convSeqWritten){
	when(convSeqCounter === (kernelSize*kernelSize-1).U){
		convSeq(convSeqCounter) := (inputBufferSize+3).U
		convSeqWritten := true.B
	}.otherwise{
		when(convSeqCounter < realKernelSize * realKernelSize){
			convSeq(convSeqCounter) := (inSize + (2*padding).U)*convSeqCounter_y + convSeqCounter_x
			when(convSeqCounter_x === realKernelSize - 1.U){
				when(convSeqCounter_y < realKernelSize - 1.U){
					convSeqCounter_x := 0.U
					convSeqCounter_y := convSeqCounter_y + 1.U
				}.otherwise{
					convSeqCounter_x := convSeqCounter_x + 1.U
				}
			}.otherwise{
				convSeqCounter_x := convSeqCounter_x + 1.U
			}
		}.otherwise{
			convSeq(convSeqCounter) := (inputBufferSize+3).U
			convSeqCounter_x := convSeqCounter_x + 1.U
		}
	}
  }
  

  // Store kernel into kernelMatrix
  val k_len = RegInit(0.U((log2Up(kernelSize*kernelSize/meshColumns)+1).W))
  val kernelStorePointer = RegInit(0.U((log2Up(kernelSize*kernelSize).W)))
  val k_len_counter = RegInit(0.U((log2Up(kernelSize*kernelSize/meshColumns)+1).W))
  when(io.k_len.fire()){
	k_len := io.k_len.bits	
  }
  when(io.k.fire() & io.lb_control & convSeqWritten){
    io.k.bits.zipWithIndex.foreach{case(k, i) => kernelMatrix(kernelStorePointer + i.U) := k(0)}
    when(k_len_counter === k_len - 1.U){
      kernelWritten := true.B
    }.otherwise{
	  k_len_counter := k_len_counter+1.U
	  kernelStorePointer := kernelStorePointer + meshColumns.U	
    }
  }
  

  // Begin to store weight activations into inputBuffer
  when(kernelWritten & io.b.fire()){
	when(bufferPointer_x === inSize + padding.U - 1.U){
		bufferPointer_x := padding.U
		when((bufferPointer_y === inSize + padding.U - 1.U) && continuous){
            bufferPointer_y := padding.U
        }.otherwise{
			bufferPointer_y := bufferPointer_y + 1.U
			// TODO: the last line padding needs to be more careful
			when(bufferPointer_y === inSize + padding.U - 1.U){
				for(i <- 0 until 12*padding){
    				inputBuffer.foreach((buff) => buff(bufferPointer(log2Up(inputBufferSize)-1,0)+(i+1).U) := (0.U).asTypeOf(inputType))
				}
			}.otherwise{
				for(i <- 0 until 2*padding){
    				inputBuffer.foreach((buff) => buff(bufferPointer(log2Up(inputBufferSize)-1,0)+(i+1).U) := (0.U).asTypeOf(inputType))
				}
			}
		}
    }.otherwise{
        bufferPointer_x := bufferPointer_x + 1.U
    }
    
    when(bufferPointer_y >= realKernelSize && bufferPointer_x >= realKernelSize){
        kernelMoveBegin := true.B
    }

    (inputBuffer zip io.b.bits).foreach{case(buff, in) => buff(bufferPointer(log2Up(inputBufferSize)-1,0)) := in(0)}
  }

  // Compute Output
  val computeMesh = Wire(Vec(meshColumns, Vec(kernelSize*kernelSize, inputType)))
  val trueKernelPointer = Wire(Vec(kernelSize*kernelSize, UInt((log2Up(inputBufferSize)+1).W)))
  (trueKernelPointer zip convSeq).foreach{case(p,s) => p := s + kernelPointer}
  computeMesh.zipWithIndex.foreach{case(permesh,z) => permesh.zipWithIndex.foreach{case(mesh, y) => 
		mesh := Mux(convSeq(y) > inputBufferSize.U, 0.S, inputBuffer(z)(trueKernelPointer(y)(log2Up(inputBufferSize)-1,0))) 
  }}
  when(kernelMoveBegin & io.out.ready){
	when(kernelPointer_x === inSize-1.U){
		kernelPointer_x := 0.U
		when(kernelPointer_y === inSize-1.U){
            kernelPointer_y := 0.U
        }.otherwise{
			kernelPointer_y :=kernelPointer_y + 1.U
		}
    }.otherwise{
        kernelPointer_x := kernelPointer_x + 1.U
    }
  }

  val vecMul = Seq.tabulate(meshColumns){m => Module(new VecMul(inputType, inputType, kernelSize*kernelSize)).io}
  vecMul.zipWithIndex.foreach{case(vec, i) => vec.in_vec_a := computeMesh(i)
											  vec.in_vec_b := kernelMatrix
											  io.out.bits(i)(0) := vec.out_c
  }
  // Connect to output
  io.out.valid := kernelMoveBegin & io.lb_control	
  
  io.b.ready := kernelWritten & io.out.ready // TODO: add queue at input and output to enable back pressure
  io.k.ready := !kernelWritten & convSeqWritten
  io.k_len.ready := !configWritten
  io.lb_config.ready := !configWritten
  
  io.tag_out:= tagReg
  // Clear states after finishing 
  when(!io.lb_control & lb_control_prev){
	inputBuffer.foreach((a) => a.foreach((b) => b := (0.U).asTypeOf(inputType)))
 	kernelMatrix.foreach((a) => a := (0.U).asTypeOf(inputType))
	bufferPointer_x := padding.U
    bufferPointer_y := padding.U
    kernelPointer_x := 0.U
    kernelPointer_y := 0.U
	convSeq.foreach((a) => a := (inputBufferSize+3).U)
	convSeqCounter_x := 0.U
    convSeqCounter_y := 0.U
    k_len := 0.U
	k_len_counter := 0.U
	kernelStorePointer := 0.U
	configWritten := false.B
    convSeqWritten := false.B
	kernelMoveBegin := false.B
    kernelWritten := false.B
    tag_written := false.B
  }

}

