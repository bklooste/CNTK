//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "SpecialPurposeNodes.h"


#include <string>
#include <vector>
#include <stdexcept>
#include <memory>



namespace Microsoft { namespace MSR { namespace CNTK {


// -----------------------------------------------------------------------
// Trace (node, say='', logFrequency=10, logFirst=10, logGradientToo=false, onlyUpToRow=100000000, onlyUpToT=100000000, format=[])
//
// Debugging aid to trace a node's value using WriteMinibatchWithFormatting().
// -----------------------------------------------------------------------

template <class ElemType>
TraceNode<ElemType>::TraceNode(const ScriptableObjects::IConfigRecordPtr configp) :
    TraceNode(configp->Get(L"deviceId"), L"<placeholder>")
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    m_message        = (const std::wstring&)configp->Get(L"say");
    m_logFirst       = configp->Get(L"logFirst");
    m_logFrequency   = configp->Get(L"logFrequency");
    m_logGradientToo = configp->Get(L"logGradientToo");
    m_formattingOptions = WriteFormattingOptions(*configp);
    m_onlyUpToRow    = configp->Get(L"onlyUpToRow");
    m_onlyUpToT      = configp->Get(L"onlyUpToT");
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Save(File& fstream) const /*override*/
{
    Base::Save(fstream);
    fstream << m_message;
    fstream << m_logFirst;
    fstream << m_logFrequency;
    fstream << m_logGradientToo;
    m_formattingOptions.Save(fstream);
    // BUGBUG: This serializes the pathname of the mapping file to disk. Not nice. But no better solution.
    fstream << m_onlyUpToRow;
    fstream << m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);
    fstream >> m_message;
    fstream >> m_logFirst;
    fstream >> m_logFrequency;
    fstream >> m_logGradientToo;
    m_formattingOptions.Load(fstream, modelVersion);
    fstream >> m_onlyUpToRow;
    fstream >> m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::BeginForwardProp() /*override*/
{
    Base::BeginForwardProp();
    ++m_numMBsRun;
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
    size_t rank = DetermineElementwiseTensorRank();
    auto result =             ValueTensorFor(rank, fr);
    auto input  = InputRef(0).ValueTensorFor(rank, fr);
    result.AssignCopyOf(input);

    // do the tracing
    Log(fr, false/*means log value*/);
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
    assert(inputIndex == 0); inputIndex;

    size_t rank = DetermineElementwiseTensorRank();
    auto sliceOutputGrad =             GradientTensorFor(rank, fr);      // propagate from this one...
    auto sliceInputGrad  = InputRef(0).GradientTensorFor(rank, fr);      // ...to this one

    sliceInputGrad.AddCopyOf(sliceOutputGrad);

    // do the tracing
    if (m_logGradientToo)
        Log(fr, true/*means log gradient*/);
}

// log value or gradient
template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Log(const FrameRange& fr, bool logGradientInstead) const
{
    if (m_numMBsRun == 1)
    {
        const auto prologue = m_formattingOptions.Processed(NodeName(), m_formattingOptions.prologue, m_numMBsRun);
        fprintf(stderr, "%s", prologue.c_str());
    }
    if (m_numMBsRun <= m_logFirst || (m_logFrequency && (m_numMBsRun-1) % m_logFrequency == 0))
    {
        char formatChar = !m_formattingOptions.isCategoryLabel ? 'f' : !m_formattingOptions.labelMappingFile.empty() ? 's' : 'u';
        auto valueFormatString = "%" + m_formattingOptions.precisionFormat + formatChar; // format string used in fprintf() for formatting the values
        const auto sequenceSeparator = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequenceSeparator, m_numMBsRun);
        const auto sequencePrologue  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequencePrologue,  m_numMBsRun);
        const auto sequenceEpilogue  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sequenceEpilogue,  m_numMBsRun);
        const auto elementSeparator  = m_formattingOptions.Processed(NodeName(), m_formattingOptions.elementSeparator,  m_numMBsRun);
        const auto sampleSeparator   = m_formattingOptions.Processed(NodeName(), m_formattingOptions.sampleSeparator,   m_numMBsRun);

        let timeRange = fr.GetTimeRange();
        fprintf(stderr, "------- Trace["); // --- for better visual separability from actual content
        if (fr.IsAllFrames())
            ;
        else if (timeRange.second == timeRange.first + 1)
            fprintf(stderr, "%d", (int)timeRange.first);
        else if (timeRange.second > timeRange.first + 1)
            fprintf(stderr, "%d..%d", (int)timeRange.first, (int)timeRange.second-1);
        fprintf(stderr, "] %ls %s--> %s\n", m_message.c_str(), logGradientInstead ? "(gradient) " : "", InputRef(0).FormatOperationPrototype("").c_str());
        InputRef(0).WriteMinibatchWithFormatting(stderr, fr, m_onlyUpToRow, m_onlyUpToT, m_formattingOptions.transpose, m_formattingOptions.isCategoryLabel, m_formattingOptions.isSparse, m_labelMapping,
                                               sequenceSeparator, sequencePrologue, sequenceEpilogue, elementSeparator, sampleSeparator,
                                               valueFormatString, logGradientInstead);
    }
}

template <class ElemType>
/*virtual*/ void TraceNode<ElemType>::Validate(bool isFinalValidationPass) // override
{
    ValidateUnaryMap(isFinalValidationPass);
    if (isFinalValidationPass)
    {
        if (m_labelMapping.empty() && (m_formattingOptions.isCategoryLabel || m_formattingOptions.isSparse) && !m_formattingOptions.labelMappingFile.empty())
            File::LoadLabelFile(m_formattingOptions.labelMappingFile, m_labelMapping);
    }
    m_numMBsRun = 0;
}

template class TraceNode<float>;
template class TraceNode<double>;

//const map<int, int> A::myMap = A::create_map();
//


template <class ElemType>
FunctionNode<ElemType>::FunctionNode(const ScriptableObjects::IConfigRecordPtr configp) :
	FunctionNode(configp->Get(L"deviceId"), L"<placeholder>")
{
	AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
	m_funcName = std::string(m_nodeName.begin(), m_nodeName.end());
	//m_message = (const std::wstring&)configp->Get(L"say");
	//m_logFirst = configp->Get(L"logFirst");
	//m_logFrequency = configp->Get(L"logFrequency");
	//m_logGradientToo = configp->Get(L"logGradientToo");
	//m_formattingOptions = WriteFormattingOptions(*configp);
	//m_onlyUpToRow = configp->Get(L"onlyUpToRow");
	//m_onlyUpToT = configp->Get(L"onlyUpToT");
}

template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::Save(File& fstream) const /*override*/
{
	Base::Save(fstream);
	//fstream << m_message;
	//fstream << m_logFirst;
	//fstream << m_logFrequency;
	//fstream << m_logGradientToo;
	//m_formattingOptions.Save(fstream);
	//// BUGBUG: This serializes the pathname of the mapping file to disk. Not nice. But no better solution.
	//fstream << m_onlyUpToRow;
	//fstream << m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
	Base::Load(fstream, modelVersion);
	//fstream >> m_message;
	//fstream >> m_logFirst;
	//fstream >> m_logFrequency;
	//fstream >> m_logGradientToo;
	//m_formattingOptions.Load(fstream, modelVersion);
	//fstream >> m_onlyUpToRow;
	//fstream >> m_onlyUpToT;
}

template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::BeginForwardProp() /*override*/
{
	Base::BeginForwardProp();
	//++m_numMBsRun;
}

template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
	size_t rank = DetermineElementwiseTensorRank();
	auto result = ValueTensorFor(rank, fr);
	auto input = InputRef(0).ValueTensorFor(rank, fr);
	FunctionNodeExternCall(input);
	result.AssignCopyOf(input);

	// do the tracing
	//Log(fr, false/*means log value*/);
}


//CHANGE THIS TO A GLOBAL LIST
template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::FunctionNodeExternCall(TensorView<ElemType>& tensor)
{
	auto fmap = get_map();
	typename map<string, externalFunc>::iterator it = fmap.find(m_funcName);

	externalFunc func;

	if (it == fmap.end())
	{
		std::string module = "./hello.so";
		//CreateDeserializerFactory f = (CreateDeserializerFactory)Plugin::Load(deserializerModule, "CreateDeserializer");
		func = (externalFunc) Plugin::Load("abc", "f");
	}
	else 
		func = it->second; 
	
	void* ptr = (void*) &tensor;
	func( ptr);
}



//externalFunc GetFunc() /*override*/
//{
//	//fmap[type] = func;     // adding the newly created Fruit to the types map for later lookup
//
//	fprintf(stdout, "C++ dlopen");
//
//
//	// open the library
//	fprintf(stdout, "Opening hello.so...\n");
//	void* handle = dlopen("./hello.so", RTLD_LAZY);
//
//
//	if (!handle) {
//	/*	auto msg = "Cannot open library" + std::to_string(dlerror());
//		fprintf(stdout, msg);
//		throw std::exception(msg);*/
//		RuntimeError("Cannot open library '%s'", dlerror());
//	}
//
//	// load the symbol
////	cout << "Loading symbol hello...\n";
//	//typedef void(*hello_t)();
//
//	// reset errors
//	dlerror();
//	TensorFunctionFunc hello = (TensorFunctionFunc)dlsym(handle, m_funcName.c_str());
//	const char *dlsym_error = dlerror();
//	if (dlsym_error) {
//		//cerr << "Cannot load symbol 'hello': " << dlsym_error <<
//		//	'\n';
//		dlclose(handle);
//		RuntimeError("FunctionNode Cannot load symbol  '%s'", dlsym_error);
//	//	throw std::exception("Cannot load symbol 'hello': " , );
//		//return 1;
//	}
//
//	// use it to do the calculation
////	cout << "Calling hello...\n";
//	//hello();
//
//	// close the library
//	fprintf(stdout, "Closing library...\n");
//	dlclose(handle);
//
//}


template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::BackpropTo(const size_t inputIndex, const FrameRange& fr) /*override*/
{
	assert(inputIndex == 0); inputIndex;

	size_t rank = DetermineElementwiseTensorRank();
	auto sliceOutputGrad = GradientTensorFor(rank, fr);      // propagate from this one...
	auto sliceInputGrad = InputRef(0).GradientTensorFor(rank, fr);      // ...to this one


	// call function with derivative appended to name

	sliceInputGrad.AddCopyOf(sliceOutputGrad);
}



template <class ElemType>
/*virtual*/ void FunctionNode<ElemType>::Validate(bool isFinalValidationPass) // override
{
	ValidateUnaryMap(isFinalValidationPass);
	//if (isFinalValidationPass)
	//{
	//	if (m_labelMapping.empty() && (m_formattingOptions.isCategoryLabel || m_formattingOptions.isSparse) && !m_formattingOptions.labelMappingFile.empty())
	//		File::LoadLabelFile(m_formattingOptions.labelMappingFile, m_labelMapping);
	//}
//	m_numMBsRun = 0;
}




template class FunctionNode<float>;
template class FunctionNode<double>;


//const FunctionNode<float>::fmap = FunctionNode<float>::create_map();
//FunctionNode<double>::fmap = FunctionNode<double>::create_map();

//
//fmap = FunctionNode::create_map();
map<string, externalFunc>& get_map()
{
	static map<string, externalFunc> functions_map;
	return functions_map;
}

}

}}
