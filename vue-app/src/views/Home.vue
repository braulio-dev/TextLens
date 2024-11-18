<script setup>
import Generator from '@/sections/Generator.vue'
import Summary from '@/sections/Summary.vue'
import { ref } from 'vue'

const summaryData = ref('')
const showSummary = ref(false)
const extractedTextData = ref('')

const blockSummary = () => {
  showSummary.value = false
}

const handleSummary = (summary, extractedText) => {
  summaryData.value = summary
  extractedTextData.value = extractedText
  showSummary.value = true
  setTimeout(() => {
    document.getElementById('summary').scrollIntoView({ behavior: 'smooth' })
  }, 100)
}
</script>

<template>
  <div :class="['flex flex-col items-center justify-center', showSummary ? 'min-h-screen' : 'h-screen']">
    <div class="flex items-center justify-center min-h-screen w-full">
      <Generator @summary-received="handleSummary" @block-summary="blockSummary" />
    </div>
    <div v-if="showSummary" class="flex items-center justify-center min-h-screen w-full">
      <Summary :summary="summaryData" :extracted_text="extractedTextData"/>
    </div>
  </div>
</template>