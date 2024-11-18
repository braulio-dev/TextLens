import { createRouter, createWebHistory } from 'vue-router'
import SPLayout from '@/layouts/SPLayout.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: SPLayout,
      children: [
        {
          path: '',
          name: 'Home',
          component: () => import('@/views/Home.vue'),
        },
      ],
    }
  ],
})

export default router
 